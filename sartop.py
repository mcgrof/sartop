#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# indent = tab
# tab-size = 4

import os, sys, threading, signal, subprocess
from time import time, sleep, strftime
from typing import List, Dict, Tuple, Optional, Any
from collections import deque
from math import ceil
import fcntl, termios, tty
from select import select
import argparse
import json
from datetime import datetime
from pathlib import Path

# Try to import matplotlib for graph generation
try:
    import matplotlib
    matplotlib.use("Agg")  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

VERSION = "1.0.0"


class Term:
    """Terminal control codes and variables"""

    width: int = 0
    height: int = 0
    resized: bool = True
    _w: int = 0
    _h: int = 0

    hide_cursor = "\033[?25l"
    show_cursor = "\033[?25h"
    alt_screen = "\033[?1049h"
    normal_screen = "\033[?1049l"
    clear = "\033[2J\033[0;0f"
    normal = "\033[0m"
    bold = "\033[1m"
    unbold = "\033[22m"
    dim = "\033[2m"
    undim = "\033[22m"

    @classmethod
    def refresh(cls):
        """Get terminal dimensions"""
        try:
            cls._w, cls._h = os.get_terminal_size()
        except:
            cls._w, cls._h = 80, 24

        if cls._w != cls.width or cls._h != cls.height:
            cls.width = cls._w
            cls.height = cls._h
            cls.resized = True


class Color:
    """Color management for terminal output"""

    @staticmethod
    def fg(r: int, g: int, b: int) -> str:
        return f"\033[38;2;{r};{g};{b}m"

    @staticmethod
    def bg(r: int, g: int, b: int) -> str:
        return f"\033[48;2;{r};{g};{b}m"

    @staticmethod
    def gradient(value: float, colors: List[Tuple[int, int, int]]) -> str:
        """Generate color based on value (0.0-1.0) across gradient"""
        if value <= 0:
            return Color.fg(*colors[0])
        if value >= 1:
            return Color.fg(*colors[-1])

        segment_size = 1.0 / (len(colors) - 1)
        segment = int(value / segment_size)
        segment_pos = (value % segment_size) / segment_size

        if segment >= len(colors) - 1:
            return Color.fg(*colors[-1])

        c1 = colors[segment]
        c2 = colors[segment + 1]

        r = int(c1[0] + (c2[0] - c1[0]) * segment_pos)
        g = int(c1[1] + (c2[1] - c1[1]) * segment_pos)
        b = int(c1[2] + (c2[2] - c1[2]) * segment_pos)

        return Color.fg(r, g, b)


class Theme:
    """Color theme definitions"""

    # CPU utilization gradient (green -> yellow -> red)
    cpu_gradient = [
        (0, 200, 0),    # Green
        (200, 200, 0),  # Yellow
        (255, 100, 0),  # Orange
        (255, 0, 0),    # Red
    ]

    # Memory gradient (blue -> cyan -> yellow -> red)
    mem_gradient = [
        (0, 100, 200),  # Blue
        (0, 200, 200),  # Cyan
        (200, 200, 0),  # Yellow
        (255, 0, 0),    # Red
    ]

    # Network/IO gradient (green -> blue -> purple)
    io_gradient = [
        (0, 255, 100),  # Green
        (0, 150, 255),  # Blue
        (150, 0, 255),  # Purple
    ]

    main_fg = Color.fg(200, 200, 200)
    main_bg = Color.bg(0, 0, 0)
    title = Color.fg(255, 255, 255)
    border = Color.fg(100, 100, 100)
    text = Color.fg(180, 180, 180)
    selected = Color.fg(0, 255, 200)


class Box:
    """Base class for UI boxes"""

    @staticmethod
    def draw(x: int, y: int, w: int, h: int, title: str = "") -> str:
        """Draw a box with optional title"""
        out = []

        # Top border
        out.append(f"\033[{y};{x}f" + Theme.border + "┌")
        if title:
            title_str = f"─┤ {Theme.title}{title}{Theme.border} ├"
            out.append(title_str)
            remaining = w - len(title) - 6
            out.append("─" * remaining)
        else:
            out.append("─" * (w - 2))
        out.append("┐")

        # Sides
        for i in range(1, h - 1):
            out.append(f"\033[{y + i};{x}f│")
            out.append(f"\033[{y + i};{x + w - 1}f│")

        # Bottom border
        out.append(f"\033[{y + h - 1};{x}f└" + "─" * (w - 2) + "┘")

        return "".join(out)


class Graph:
    """Graph drawing for metrics visualization"""

    def __init__(self, width: int, height: int, max_value: float = 100.0):
        self.width = width
        self.height = height
        self.max_value = max_value
        self.data = deque(maxlen=width)

    def add_value(self, value: float):
        """Add a new value to the graph"""
        self.data.append(min(value, self.max_value))

    def draw(self, x: int, y: int, gradient: List[Tuple[int, int, int]]) -> str:
        """Draw the graph at position x, y"""
        out = []

        # Fill with zeros if not enough data
        data_list = list(self.data)
        while len(data_list) < self.width:
            data_list.insert(0, 0)

        # Create a 2D grid to represent the graph
        grid = [[" " for _ in range(self.width)] for _ in range(self.height)]

        # Plot each data point as a small dot
        for col, value in enumerate(data_list):
            # Calculate which row this value should appear on
            graph_height = (value / self.max_value) * self.height
            row_from_top = self.height - int(graph_height)

            # Ensure we're within bounds
            if row_from_top < 0:
                row_from_top = 0
            elif row_from_top >= self.height:
                row_from_top = self.height - 1

            # Use small dot character
            grid[row_from_top][col] = "·"

        # Render the grid with colors
        for row in range(self.height):
            out.append(f"\033[{y + row};{x}f")
            for col in range(self.width):
                if grid[row][col] != " ":
                    # Calculate value for this position for coloring
                    value = data_list[col] if col < len(data_list) else 0
                    color = Color.gradient(value / self.max_value, gradient)
                    out.append(color + grid[row][col])
                else:
                    out.append(" ")
            out.append(Term.normal)

        return "".join(out)


class DualDirectionGraph:
    """Dual-direction graph showing reads (up) and writes (down) from center line"""

    def __init__(self, width: int, height: int, max_value: float = 10000.0):
        self.width = width
        self.height = height
        self.max_value = max_value
        self.read_data = deque(maxlen=width)
        self.write_data = deque(maxlen=width)
        self.center_line = height // 2

    def add_values(self, read: float, write: float):
        """Add new read and write values"""
        self.read_data.append(min(read, self.max_value))
        self.write_data.append(min(write, self.max_value))

    def draw(self, x: int, y: int, gradient: List[Tuple[int, int, int]]) -> str:
        """Draw dual-direction graph with reads above center, writes below"""
        out = []

        # Prepare data lists
        read_list = list(self.read_data)
        write_list = list(self.write_data)
        while len(read_list) < self.width:
            read_list.insert(0, 0)
            write_list.insert(0, 0)

        # Create grid
        grid = [[" " for _ in range(self.width)] for _ in range(self.height)]
        colors = [[(0, 0, 0) for _ in range(self.width)] for _ in range(self.height)]

        # Draw center line
        for col in range(self.width):
            grid[self.center_line][col] = "─"
            colors[self.center_line][col] = (100, 100, 100)

        # Plot reads (going upward from center)
        for col, value in enumerate(read_list):
            if value > 0:
                intensity = value / self.max_value
                bar_height = int((intensity * self.center_line))
                for row in range(max(0, self.center_line - bar_height), self.center_line):
                    grid[row][col] = "│"
                    colors[row][col] = intensity

        # Plot writes (going downward from center)
        for col, value in enumerate(write_list):
            if value > 0:
                intensity = value / self.max_value
                bar_height = int((intensity * (self.height - self.center_line)))
                for row in range(self.center_line + 1, min(self.height, self.center_line + bar_height + 1)):
                    grid[row][col] = "│"
                    colors[row][col] = intensity

        # Render with colors
        for row in range(self.height):
            out.append(f"\033[{y + row};{x}f")
            for col in range(self.width):
                char = grid[row][col]
                if char == "─":
                    out.append(Color.fg(100, 100, 100) + char)
                elif char == "│":
                    color = Color.gradient(colors[row][col], gradient)
                    out.append(color + char)
                else:
                    out.append(" ")
            out.append(Term.normal)

        return "".join(out)


class IntensityMeter:
    """Unicode block-based intensity meter for IOPS visualization"""

    BLOCKS = " ▁▂▃▄▅▆▇█"

    @staticmethod
    def draw(x: int, y: int, width: int, value: float, max_value: float,
             label: str = "", gradient: List[Tuple[int, int, int]] = None) -> str:
        """Draw horizontal intensity meter using Unicode blocks"""
        out = []
        if gradient is None:
            gradient = Theme.io_gradient

        # Calculate intensity (0.0 to 1.0)
        intensity = min(value / max_value, 1.0) if max_value > 0 else 0.0

        # Draw label
        if label:
            out.append(f"\033[{y};{x}f{Theme.text}{label}: ")
            label_len = len(label) + 2
        else:
            out.append(f"\033[{y};{x}f")
            label_len = 0

        # Calculate filled width
        filled_width = int(intensity * width)

        # Draw filled blocks
        color = Color.gradient(intensity, gradient)
        for i in range(filled_width):
            block_intensity = min((i + 1) / width, 1.0)
            block_idx = int(block_intensity * (len(IntensityMeter.BLOCKS) - 1))
            out.append(color + IntensityMeter.BLOCKS[block_idx])

        # Draw empty blocks
        for i in range(filled_width, width):
            out.append(Theme.border + IntensityMeter.BLOCKS[0])

        # Draw value
        out.append(f" {Theme.selected}{value:.1f}")
        out.append(Term.normal)

        return "".join(out)


class UtilizationBar:
    """Horizontal bar showing device utilization percentage"""

    @staticmethod
    def draw(x: int, y: int, width: int, util_percent: float, label: str = "") -> str:
        """Draw utilization bar with color coding"""
        out = []

        # Label
        if label:
            out.append(f"\033[{y};{x}f{Theme.text}{label}: ")
            label_len = len(label) + 2
        else:
            out.append(f"\033[{y};{x}f")
            label_len = 0

        # Color based on utilization
        if util_percent < 50:
            color = Color.fg(0, 200, 0)  # Green
        elif util_percent < 80:
            color = Color.fg(200, 200, 0)  # Yellow
        else:
            color = Color.fg(255, 0, 0)  # Red

        # Draw bar
        filled = int((util_percent / 100.0) * width)
        out.append(color + "█" * filled)
        out.append(Theme.border + "░" * (width - filled))
        out.append(f" {Theme.selected}{util_percent:.1f}%")
        out.append(Term.normal)

        return "".join(out)


class LatencyGauge:
    """Visual latency indicator with color-coded warnings"""

    @staticmethod
    def draw(x: int, y: int, await_ms: float, label: str = "") -> str:
        """Draw latency gauge with color coding (green < 10ms, yellow < 50ms, red >= 50ms)"""
        out = []

        # Determine color and status
        if await_ms < 10:
            color = Color.fg(0, 255, 0)  # Green
            status = "●"
            status_text = "FAST"
        elif await_ms < 50:
            color = Color.fg(255, 255, 0)  # Yellow
            status = "◐"
            status_text = "MODERATE"
        else:
            color = Color.fg(255, 0, 0)  # Red
            status = "●"
            status_text = "SLOW"

        # Draw
        out.append(f"\033[{y};{x}f{Theme.text}{label}: {color}{status} {await_ms:.2f}ms {Term.dim}({status_text})")
        out.append(Term.normal)

        return "".join(out)


class DiskHeatmap:
    """Timeline heatmap showing I/O activity intensity over time"""

    def __init__(self, width: int):
        self.width = width
        self.history = deque(maxlen=width)

    def add_value(self, value: float):
        """Add a new intensity value (0.0 to 1.0)"""
        self.history.append(min(value, 1.0))

    def draw(self, x: int, y: int, gradient: List[Tuple[int, int, int]]) -> str:
        """Draw heatmap timeline using background colors"""
        out = [f"\033[{y};{x}f"]

        # Fill with zeros if needed
        data_list = list(self.history)
        while len(data_list) < self.width:
            data_list.insert(0, 0)

        # Draw each cell with background color
        for value in data_list:
            if value < 0.1:
                # Very low activity - dark
                out.append(Color.bg(20, 20, 20) + " ")
            else:
                # Scale to gradient
                color_val = Color.gradient(value, gradient)
                # Extract RGB from color string (hacky but works)
                out.append(Color.bg(
                    int(50 + value * 205),
                    int(50 + value * 100),
                    int(50)
                ) + " ")

        out.append(Term.normal)
        return "".join(out)


class SARCollector:
    """Collects SAR data using sar command"""

    def __init__(self):
        self.process = None
        self.data_lock = threading.Lock()
        self.latest_data = {}
        self.running = False

    def start(self):
        """Start collecting SAR data"""
        self.running = True
        self.thread = threading.Thread(target=self._collect_loop, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop collecting SAR data"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)

    def _collect_loop(self):
        """Main collection loop"""
        while self.running:
            try:
                # Run sar command to get current stats
                # -u: CPU, -r: Memory, -n DEV: Network, -d: Disk
                result = subprocess.run(
                    ['sar', '-u', '-r', '-n', 'DEV', '-d', '1', '1'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )

                if result.returncode == 0:
                    data = self._parse_sar_output(result.stdout)
                    with self.data_lock:
                        self.latest_data = data

            except Exception as e:
                # Silently continue on errors
                pass

            sleep(1)

    def _parse_sar_output(self, output: str) -> Dict[str, Any]:
        """Parse SAR text output into structured data"""
        data = {
            'cpu': {'user': 0, 'system': 0, 'iowait': 0, 'idle': 100},
            'memory': {'used_percent': 0, 'buffers': 0, 'cached': 0},
            'network': {},
            'disk': {}
        }

        lines = output.split('\n')
        section = None

        for line in lines:
            line = line.strip()
            if not line or line.startswith('Linux'):
                continue

            # Detect section headers
            if 'CPU' in line and '%user' in line:
                section = 'cpu'
                continue
            elif 'kbmemfree' in line or 'memfree' in line:
                section = 'memory'
                continue
            elif 'IFACE' in line or 'iface' in line:
                section = 'network'
                continue
            elif 'DEV' in line and 'tps' in line:
                section = 'disk'
                continue

            # Skip Average lines for network/disk
            if line.startswith('Average:') and section in ['network', 'disk']:
                continue

            # Parse data based on section
            if section == 'cpu' and 'Average:' in line:
                parts = line.split()
                if len(parts) >= 7:
                    try:
                        data['cpu'] = {
                            'user': float(parts[2]),
                            'system': float(parts[4]),
                            'iowait': float(parts[5]),
                            'idle': float(parts[7])
                        }
                    except:
                        pass

            elif section == 'memory' and 'Average:' in line:
                parts = line.split()
                if len(parts) >= 5:
                    try:
                        # Try to find the %memused field
                        if '%memused' in output:
                            idx = 0
                            for i, part in enumerate(parts):
                                if part == 'Average:':
                                    idx = i
                                    break
                            # %memused is typically around position 4-5
                            if len(parts) > idx + 4:
                                data['memory']['used_percent'] = float(parts[idx + 4])
                    except:
                        pass

            elif section == 'network':
                parts = line.split()
                if len(parts) >= 9 and parts[0] not in ['Average:', 'IFACE', 'iface']:
                    try:
                        # Time IFACE rxpck/s txpck/s rxkB/s txkB/s ...
                        iface = parts[1]
                        data['network'][iface] = {
                            'rxkB': float(parts[4]),
                            'txkB': float(parts[5])
                        }
                    except:
                        pass

            elif section == 'disk':
                parts = line.split()
                if len(parts) >= 11 and parts[0] not in ['Average:', 'DEV']:
                    try:
                        # Format: Time PM DEV tps rkB/s wkB/s dkB/s areq-sz aqu-sz await %util
                        # Or:     Time AM DEV tps rkB/s wkB/s dkB/s areq-sz aqu-sz await %util
                        dev = parts[2]  # Device name is at position 2
                        data['disk'][dev] = {
                            'tps': float(parts[3]),
                            'rkB': float(parts[4]),
                            'wkB': float(parts[5]),
                            'dkB': float(parts[6]),
                            'areq_sz': float(parts[7]),
                            'aqu_sz': float(parts[8]),
                            'await': float(parts[9]),
                            'util': float(parts[10])
                        }
                    except:
                        pass

        return data

    def get_latest(self) -> Dict[str, Any]:
        """Get latest collected data"""
        with self.data_lock:
            return self.latest_data.copy()


class SARTop:
    """Main application class"""

    def __init__(self, interval: int = 1):
        self.interval = interval
        self.collector = SARCollector()
        self.running = False

        # Data storage
        self.history = []
        self.start_time = None
        self.base_filename = None  # Will be set when saving

        # Graphs
        graph_width = 60
        graph_height = 8

        self.cpu_graph = Graph(graph_width, graph_height, 100)
        self.mem_graph = Graph(graph_width, graph_height, 100)
        self.net_rx_graph = Graph(graph_width, graph_height, 10000)  # kB/s
        self.net_tx_graph = Graph(graph_width, graph_height, 10000)  # kB/s

        # Disk I/O graphs and visualizers
        self.disk_io_graph = DualDirectionGraph(graph_width, graph_height, 50000)  # kB/s
        self.disk_iops_graph = Graph(graph_width, graph_height, 1000)  # tps
        self.disk_util_heatmap = DiskHeatmap(graph_width)
        self.disk_latency_graph = Graph(graph_width, graph_height, 100)  # ms

        # Statistics
        self.stats = {
            'cpu': {'avg': 0, 'max': 0, 'min': 100},
            'memory': {'avg': 0, 'max': 0, 'min': 100},
            'network_rx': {'avg': 0, 'max': 0},
            'network_tx': {'avg': 0, 'max': 0},
            'disk': {
                'read_kBps': {'avg': 0, 'max': 0, 'total': 0},
                'write_kBps': {'avg': 0, 'max': 0, 'total': 0},
                'tps': {'avg': 0, 'max': 0},
                'await': {'avg': 0, 'max': 0, 'min': 999999},
                'util': {'avg': 0, 'max': 0},
                'per_device': {}  # Stats per device
            }
        }

        # Disk I/O display mode (0=Overview, 1=Detailed, 2=Heatmap, 3=IOPS)
        self.disk_mode = 0

    def start(self):
        """Start monitoring"""
        self.running = True
        self.start_time = datetime.now()

        # Setup terminal
        print(Term.alt_screen + Term.hide_cursor + Term.clear, end='', flush=True)

        # Start SAR collector
        self.collector.start()

        # Main loop
        try:
            while self.running:
                self._update()
                sleep(self.interval)
        except KeyboardInterrupt:
            pass
        finally:
            self._cleanup()

    def _update(self):
        """Update display and collect data"""
        # Get latest SAR data
        data = self.collector.get_latest()

        if not data:
            return

        # Add timestamp
        data['timestamp'] = datetime.now().isoformat()

        # Store in history
        self.history.append(data)

        # Update graphs
        cpu_used = 100 - data['cpu']['idle']
        self.cpu_graph.add_value(cpu_used)

        mem_used = data['memory'].get('used_percent', 0)
        self.mem_graph.add_value(mem_used)

        # Sum network traffic across all interfaces
        total_rx = sum(iface['rxkB'] for iface in data.get('network', {}).values())
        total_tx = sum(iface['txkB'] for iface in data.get('network', {}).values())

        self.net_rx_graph.add_value(total_rx)
        self.net_tx_graph.add_value(total_tx)

        # Sum disk I/O across all devices
        disk_data = data.get('disk', {})
        total_read = sum(dev['rkB'] for dev in disk_data.values())
        total_write = sum(dev['wkB'] for dev in disk_data.values())
        total_tps = sum(dev['tps'] for dev in disk_data.values())

        # Calculate average latency and max utilization across devices
        avg_await = sum(dev['await'] for dev in disk_data.values()) / len(disk_data) if disk_data else 0
        max_util = max((dev['util'] for dev in disk_data.values()), default=0)

        # Update disk I/O graphs
        self.disk_io_graph.add_values(total_read, total_write)
        self.disk_iops_graph.add_value(total_tps)
        self.disk_util_heatmap.add_value(max_util / 100.0)
        self.disk_latency_graph.add_value(avg_await)

        # Update statistics
        self._update_stats(cpu_used, mem_used, total_rx, total_tx,
                          total_read, total_write, total_tps, avg_await, max_util, disk_data)

        # Refresh terminal size
        Term.refresh()

        # Draw UI
        self._draw()

    def _update_stats(self, cpu: float, mem: float, rx: float, tx: float,
                     disk_read: float, disk_write: float, disk_tps: float,
                     disk_await: float, disk_util: float, disk_devices: Dict):
        """Update running statistics"""
        if len(self.history) == 0:
            return

        # CPU stats
        self.stats['cpu']['max'] = max(self.stats['cpu']['max'], cpu)
        self.stats['cpu']['min'] = min(self.stats['cpu']['min'], cpu)
        self.stats['cpu']['avg'] = sum(100 - d['cpu']['idle'] for d in self.history) / len(self.history)

        # Memory stats
        self.stats['memory']['max'] = max(self.stats['memory']['max'], mem)
        self.stats['memory']['min'] = min(self.stats['memory']['min'], mem)
        self.stats['memory']['avg'] = sum(d['memory'].get('used_percent', 0) for d in self.history) / len(self.history)

        # Network stats
        self.stats['network_rx']['max'] = max(self.stats['network_rx']['max'], rx)
        self.stats['network_tx']['max'] = max(self.stats['network_tx']['max'], tx)

        # Disk I/O stats
        self.stats['disk']['read_kBps']['max'] = max(self.stats['disk']['read_kBps']['max'], disk_read)
        self.stats['disk']['write_kBps']['max'] = max(self.stats['disk']['write_kBps']['max'], disk_write)
        self.stats['disk']['tps']['max'] = max(self.stats['disk']['tps']['max'], disk_tps)
        self.stats['disk']['await']['max'] = max(self.stats['disk']['await']['max'], disk_await)
        self.stats['disk']['await']['min'] = min(self.stats['disk']['await']['min'], disk_await) if disk_await > 0 else self.stats['disk']['await']['min']
        self.stats['disk']['util']['max'] = max(self.stats['disk']['util']['max'], disk_util)

        # Calculate averages from history
        total_samples = len(self.history)
        total_read = 0
        total_write = 0
        total_tps = 0
        total_await = 0
        total_util = 0
        await_count = 0

        for d in self.history:
            disk_data = d.get('disk', {})
            total_read += sum(dev['rkB'] for dev in disk_data.values())
            total_write += sum(dev['wkB'] for dev in disk_data.values())
            total_tps += sum(dev['tps'] for dev in disk_data.values())

            if disk_data:
                avg_await = sum(dev['await'] for dev in disk_data.values()) / len(disk_data)
                total_await += avg_await
                await_count += 1
                total_util += max((dev['util'] for dev in disk_data.values()), default=0)

        self.stats['disk']['read_kBps']['avg'] = total_read / total_samples
        self.stats['disk']['write_kBps']['avg'] = total_write / total_samples
        self.stats['disk']['tps']['avg'] = total_tps / total_samples
        self.stats['disk']['await']['avg'] = total_await / await_count if await_count > 0 else 0
        self.stats['disk']['util']['avg'] = total_util / total_samples

        # Per-device statistics
        for dev_name, dev_data in disk_devices.items():
            if dev_name not in self.stats['disk']['per_device']:
                self.stats['disk']['per_device'][dev_name] = {
                    'read_max': 0, 'write_max': 0, 'tps_max': 0,
                    'await_max': 0, 'util_max': 0, 'samples': 0
                }

            dev_stats = self.stats['disk']['per_device'][dev_name]
            dev_stats['read_max'] = max(dev_stats['read_max'], dev_data['rkB'])
            dev_stats['write_max'] = max(dev_stats['write_max'], dev_data['wkB'])
            dev_stats['tps_max'] = max(dev_stats['tps_max'], dev_data['tps'])
            dev_stats['await_max'] = max(dev_stats['await_max'], dev_data['await'])
            dev_stats['util_max'] = max(dev_stats['util_max'], dev_data['util'])
            dev_stats['samples'] += 1

    def _draw(self):
        """Draw the UI"""
        out = [Term.clear]

        # Title
        title = f"SARTop v{VERSION} - System Activity Monitor"
        duration = (datetime.now() - self.start_time).total_seconds()
        duration_str = f"Duration: {int(duration)}s | Samples: {len(self.history)}"

        out.append(f"\033[1;1f{Theme.title}{Term.bold}{title}{Term.normal}")
        out.append(f"\033[1;{Term.width - len(duration_str)}f{Theme.text}{duration_str}")

        # Layout: 2 columns
        col1_x = 2
        col2_x = Term.width // 2 + 1
        box_width = Term.width // 2 - 3
        box_height = 12

        y = 3

        # CPU Box
        out.append(Box.draw(col1_x, y, box_width, box_height, "CPU Usage"))

        latest = self.history[-1] if self.history else {'cpu': {'user': 0, 'system': 0, 'idle': 100}}
        cpu_used = 100 - latest['cpu']['idle']

        out.append(f"\033[{y+1};{col1_x+2}f{Theme.text}User: {Theme.selected}{latest['cpu']['user']:.1f}%")
        out.append(f"\033[{y+2};{col1_x+2}f{Theme.text}System: {Theme.selected}{latest['cpu']['system']:.1f}%")
        out.append(f"\033[{y+2};{col1_x+25}f{Theme.text}IO Wait: {Theme.selected}{latest['cpu']['iowait']:.1f}%")

        # CPU Graph
        graph_x = col1_x + 2
        graph_y = y + 3
        out.append(self.cpu_graph.draw(graph_x, graph_y, Theme.cpu_gradient))

        # Memory Box
        out.append(Box.draw(col2_x, y, box_width, box_height, "Memory Usage"))

        mem_used = latest['memory'].get('used_percent', 0)
        out.append(f"\033[{y+1};{col2_x+2}f{Theme.text}Used: {Theme.selected}{mem_used:.1f}%")
        out.append(f"\033[{y+2};{col2_x+2}f{Theme.text}Avg: {Theme.selected}{self.stats['memory']['avg']:.1f}%")
        out.append(f"\033[{y+2};{col2_x+25}f{Theme.text}Max: {Theme.selected}{self.stats['memory']['max']:.1f}%")

        # Memory Graph
        graph_x = col2_x + 2
        graph_y = y + 3
        out.append(self.mem_graph.draw(graph_x, graph_y, Theme.mem_gradient))

        # Network RX Box
        y += box_height + 1
        out.append(Box.draw(col1_x, y, box_width, box_height, "Network RX"))

        total_rx = sum(iface['rxkB'] for iface in latest.get('network', {}).values())
        out.append(f"\033[{y+1};{col1_x+2}f{Theme.text}Current: {Theme.selected}{total_rx:.1f} kB/s")
        out.append(f"\033[{y+2};{col1_x+2}f{Theme.text}Peak: {Theme.selected}{self.stats['network_rx']['max']:.1f} kB/s")

        # Network RX Graph
        graph_x = col1_x + 2
        graph_y = y + 3
        out.append(self.net_rx_graph.draw(graph_x, graph_y, Theme.io_gradient))

        # Network TX Box
        out.append(Box.draw(col2_x, y, box_width, box_height, "Network TX"))

        total_tx = sum(iface['txkB'] for iface in latest.get('network', {}).values())
        out.append(f"\033[{y+1};{col2_x+2}f{Theme.text}Current: {Theme.selected}{total_tx:.1f} kB/s")
        out.append(f"\033[{y+2};{col2_x+2}f{Theme.text}Peak: {Theme.selected}{self.stats['network_tx']['max']:.1f} kB/s")

        # Network TX Graph
        graph_x = col2_x + 2
        graph_y = y + 3
        out.append(self.net_tx_graph.draw(graph_x, graph_y, Theme.io_gradient))

        # ============================================================
        # DISK I/O SECTION - Comprehensive visualizations
        # ============================================================
        y += box_height + 1

        # Get disk data
        disk_data = latest.get('disk', {})
        total_read = sum(dev['rkB'] for dev in disk_data.values())
        total_write = sum(dev['wkB'] for dev in disk_data.values())
        total_tps = sum(dev['tps'] for dev in disk_data.values())

        # Main Disk I/O Box - Dual Direction Graph
        disk_box_height = 14
        disk_box_width = Term.width - 4
        out.append(Box.draw(2, y, disk_box_width, disk_box_height,
                           f"Disk I/O - Read/Write Activity (Read: ↑ {total_read:.1f} kB/s | Write: ↓ {total_write:.1f} kB/s)"))

        # Draw dual-direction graph
        graph_x = 4
        graph_y = y + 2
        out.append(self.disk_io_graph.draw(graph_x, graph_y, Theme.io_gradient))

        # Add Read/Write labels
        out.append(f"\033[{y+1};{4}f{Theme.text}Read ↑")
        out.append(f"\033[{y+disk_box_height-2};{4}f{Theme.text}Write ↓")

        # Stats on the right side of the graph
        stats_x = graph_x + 62
        out.append(f"\033[{y+2};{stats_x}f{Theme.text}Total I/O")
        out.append(f"\033[{y+3};{stats_x}f{Theme.text}Read:  {Theme.selected}{self.stats['disk']['read_kBps']['avg']:.1f} {Theme.text}kB/s avg")
        out.append(f"\033[{y+4};{stats_x}f{Theme.text}       {Theme.selected}{self.stats['disk']['read_kBps']['max']:.1f} {Theme.text}kB/s peak")
        out.append(f"\033[{y+5};{stats_x}f{Theme.text}Write: {Theme.selected}{self.stats['disk']['write_kBps']['avg']:.1f} {Theme.text}kB/s avg")
        out.append(f"\033[{y+6};{stats_x}f{Theme.text}       {Theme.selected}{self.stats['disk']['write_kBps']['max']:.1f} {Theme.text}kB/s peak")

        out.append(f"\033[{y+8};{stats_x}f{Theme.text}IOPS: {Theme.selected}{total_tps:.1f} {Theme.text}tps")
        out.append(f"\033[{y+9};{stats_x}f{Theme.text}Peak: {Theme.selected}{self.stats['disk']['tps']['max']:.1f} {Theme.text}tps")

        # Per-Device Metrics Section
        y += disk_box_height + 1

        if disk_data:
            # Calculate layout for per-device section
            num_devices = len(disk_data)
            device_box_height = 8 + num_devices
            out.append(Box.draw(2, y, disk_box_width, device_box_height,
                               f"Per-Device Metrics ({num_devices} device{'s' if num_devices != 1 else ''})"))

            dev_y = y + 1
            for dev_name, dev_metrics in sorted(disk_data.items()):
                dev_y += 1

                # Device name and key metrics
                out.append(f"\033[{dev_y};{4}f{Theme.title}{Term.bold}{dev_name}{Term.normal}")

                # IOPS Intensity Meter
                out.append(IntensityMeter.draw(6, dev_y + 1, 30, dev_metrics['tps'],
                                              self.stats['disk']['tps']['max'] or 100,
                                              "IOPS", Theme.io_gradient))

                # Utilization Bar
                out.append(UtilizationBar.draw(50, dev_y + 1, 25, dev_metrics['util'], "Util"))

                # Read/Write rates
                out.append(f"\033[{dev_y + 1};{90}f{Theme.text}R: {Theme.selected}{dev_metrics['rkB']:.1f} {Theme.text}kB/s")
                out.append(f"\033[{dev_y + 1};{110}f{Theme.text}W: {Theme.selected}{dev_metrics['wkB']:.1f} {Theme.text}kB/s")

                # Latency Gauge
                out.append(LatencyGauge.draw(6, dev_y + 2, dev_metrics['await'], "Latency"))

                # Queue size and request size
                out.append(f"\033[{dev_y + 2};{50}f{Theme.text}Queue: {Theme.selected}{dev_metrics['aqu_sz']:.2f}")
                out.append(f"\033[{dev_y + 2};{70}f{Theme.text}Avg Req: {Theme.selected}{dev_metrics['areq_sz']:.1f} {Theme.text}kB")

                # Discard rate (for SSDs)
                if dev_metrics['dkB'] > 0:
                    out.append(f"\033[{dev_y + 2};{95}f{Theme.text}Discard: {Theme.selected}{dev_metrics['dkB']:.1f} {Theme.text}kB/s")

                dev_y += 2

        # Instructions at bottom
        y = Term.height - 1
        instructions = "Press Ctrl+C to exit and save data"
        out.append(f"\033[{y};{(Term.width - len(instructions)) // 2}f{Theme.border}{instructions}")

        # Flush output
        print("".join(out), end='', flush=True)

    def _cleanup(self):
        """Cleanup and save data"""
        # Stop collector
        self.collector.stop()

        # Restore terminal
        print(Term.normal_screen + Term.show_cursor, end='', flush=True)

        # Save data
        if self.history:
            self._save_json()
            if MATPLOTLIB_AVAILABLE:
                self._save_plot()
            self._save_summary()

    def _save_json(self):
        """Save collected data to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        # Use hostname for system identifier, similar to gputop's vendor name
        hostname = os.uname().nodename.split('.')[0]
        self.base_filename = f"sartop-{hostname}-{timestamp}"
        filename = f"{self.base_filename}.json"

        output_data = {
            'start_time': self.start_time.isoformat(),
            'end_time': datetime.now().isoformat(),
            'duration_seconds': (datetime.now() - self.start_time).total_seconds(),
            'sample_count': len(self.history),
            'hostname': hostname,
            'statistics': self.stats,
            'data': self.history
        }

        with open(filename, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"\nData saved to {filename}")

    def _save_plot(self):
        """Generate and save comprehensive PNG plots with disk I/O"""
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        if not self.base_filename:
            hostname = os.uname().nodename.split('.')[0]
            self.base_filename = f"sartop-{hostname}-{timestamp}"

        filename = f"{self.base_filename}_plot.png"

        # Prepare data
        times = [datetime.fromisoformat(d['timestamp']) for d in self.history]
        cpu_data = [100 - d['cpu']['idle'] for d in self.history]
        mem_data = [d['memory'].get('used_percent', 0) for d in self.history]
        rx_data = [sum(iface['rxkB'] for iface in d.get('network', {}).values()) for d in self.history]
        tx_data = [sum(iface['txkB'] for iface in d.get('network', {}).values()) for d in self.history]

        # Disk I/O data
        disk_read_data = [sum(dev['rkB'] for dev in d.get('disk', {}).values()) for d in self.history]
        disk_write_data = [sum(dev['wkB'] for dev in d.get('disk', {}).values()) for d in self.history]
        disk_tps_data = [sum(dev['tps'] for dev in d.get('disk', {}).values()) for d in self.history]
        disk_await_data = [sum(dev['await'] for dev in d.get('disk', {}).values()) / len(d.get('disk', {}))
                          if d.get('disk', {}) else 0 for d in self.history]
        disk_util_data = [max((dev['util'] for dev in d.get('disk', {}).values()), default=0) for d in self.history]

        # Create figure with 3x3 subplots for comprehensive visualization
        fig = plt.figure(figsize=(20, 14))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        fig.suptitle(f'System Performance Analysis - {timestamp}', fontsize=18, fontweight='bold')

        # Row 1: CPU and Memory
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(times, cpu_data, 'r-', linewidth=1.5, label='CPU Usage')
        ax1.fill_between(times, cpu_data, alpha=0.3, color='red')
        ax1.set_title('CPU Usage', fontweight='bold')
        ax1.set_ylabel('Usage (%)')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 100)
        ax1.legend()

        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(times, mem_data, 'b-', linewidth=1.5, label='Memory Usage')
        ax2.fill_between(times, mem_data, alpha=0.3, color='blue')
        ax2.set_title('Memory Usage', fontweight='bold')
        ax2.set_ylabel('Usage (%)')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 100)
        ax2.legend()

        # Row 1, Col 3: Disk IOPS
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(times, disk_tps_data, color='purple', linewidth=1.5, label='IOPS')
        ax3.fill_between(times, disk_tps_data, alpha=0.3, color='purple')
        ax3.set_title('Disk IOPS (Transactions/sec)', fontweight='bold')
        ax3.set_ylabel('TPS')
        ax3.grid(True, alpha=0.3)
        ax3.legend()

        # Row 2: Network and Disk I/O Bandwidth
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.plot(times, rx_data, 'g-', linewidth=1.5, label='RX')
        ax4.plot(times, tx_data, 'm-', linewidth=1.5, label='TX')
        ax4.fill_between(times, rx_data, alpha=0.2, color='green')
        ax4.fill_between(times, tx_data, alpha=0.2, color='magenta')
        ax4.set_title('Network Traffic', fontweight='bold')
        ax4.set_ylabel('kB/s')
        ax4.grid(True, alpha=0.3)
        ax4.legend()

        ax5 = fig.add_subplot(gs[1, 1])
        ax5.plot(times, disk_read_data, color='#00AA00', linewidth=1.5, label='Read')
        ax5.plot(times, disk_write_data, color='#FF6600', linewidth=1.5, label='Write')
        ax5.fill_between(times, disk_read_data, alpha=0.3, color='#00AA00')
        ax5.fill_between(times, disk_write_data, alpha=0.3, color='#FF6600')
        ax5.set_title('Disk I/O Bandwidth', fontweight='bold')
        ax5.set_ylabel('kB/s')
        ax5.grid(True, alpha=0.3)
        ax5.legend()

        # Row 2, Col 3: Disk Utilization
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.plot(times, disk_util_data, color='#FF0066', linewidth=1.5, label='Max Util')
        ax6.fill_between(times, disk_util_data, alpha=0.3, color='#FF0066')
        ax6.set_title('Disk Utilization', fontweight='bold')
        ax6.set_ylabel('Utilization (%)')
        ax6.grid(True, alpha=0.3)
        ax6.set_ylim(0, 100)
        ax6.legend()

        # Row 3: Disk Latency and Combined View
        ax7 = fig.add_subplot(gs[2, 0])
        ax7.plot(times, disk_await_data, color='#FF9900', linewidth=1.5, label='Avg Latency')
        ax7.fill_between(times, disk_await_data, alpha=0.3, color='#FF9900')
        ax7.set_title('Disk I/O Latency (await)', fontweight='bold')
        ax7.set_ylabel('Milliseconds')
        ax7.grid(True, alpha=0.3)
        ax7.legend()

        # Row 3, Col 2-3: Stacked area chart for comprehensive I/O view
        ax8 = fig.add_subplot(gs[2, 1:])
        ax8.stackplot(times, disk_read_data, disk_write_data,
                     labels=['Read', 'Write'],
                     colors=['#00DD00', '#DD6600'],
                     alpha=0.7)
        ax8.set_title('Disk I/O - Stacked View', fontweight='bold')
        ax8.set_ylabel('kB/s')
        ax8.grid(True, alpha=0.3)
        ax8.legend(loc='upper left')

        # Format x-axis for all plots
        for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            ax.set_xlabel('Time')

        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Plot saved to {filename}")

    def _save_summary(self):
        """Save and print summary statistics"""
        if not self.base_filename:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            hostname = os.uname().nodename.split('.')[0]
            self.base_filename = f"sartop-{hostname}-{timestamp}"

        filename = f"{self.base_filename}_summary.txt"
        duration = (datetime.now() - self.start_time).total_seconds()

        # Build summary content
        summary_lines = []
        summary_lines.append("System Performance Summary")
        summary_lines.append("=" * 60)
        summary_lines.append("")
        summary_lines.append(f"Hostname: {os.uname().nodename}")
        summary_lines.append(f"Duration: {int(duration)} seconds")
        summary_lines.append(f"Samples: {len(self.history)}")
        summary_lines.append(f"Start: {self.start_time.isoformat()}")
        summary_lines.append(f"End: {datetime.now().isoformat()}")
        summary_lines.append("")
        summary_lines.append("CPU Usage:")
        summary_lines.append(f"  Average: {self.stats['cpu']['avg']:.1f}%")
        summary_lines.append(f"  Max: {self.stats['cpu']['max']:.1f}%")
        summary_lines.append(f"  Min: {self.stats['cpu']['min']:.1f}%")
        summary_lines.append("")
        summary_lines.append("Memory Usage:")
        summary_lines.append(f"  Average: {self.stats['memory']['avg']:.1f}%")
        summary_lines.append(f"  Max: {self.stats['memory']['max']:.1f}%")
        summary_lines.append(f"  Min: {self.stats['memory']['min']:.1f}%")
        summary_lines.append("")
        summary_lines.append("Network Traffic:")
        summary_lines.append(f"  Peak RX: {self.stats['network_rx']['max']:.1f} kB/s")
        summary_lines.append(f"  Peak TX: {self.stats['network_tx']['max']:.1f} kB/s")
        summary_lines.append("")
        summary_lines.append("=" * 60)
        summary_lines.append("Disk I/O Performance Analysis")
        summary_lines.append("=" * 60)
        summary_lines.append("")
        summary_lines.append("Aggregate Disk I/O:")
        summary_lines.append("  Read Bandwidth:")
        summary_lines.append(f"    Average: {self.stats['disk']['read_kBps']['avg']:.1f} kB/s")
        summary_lines.append(f"    Peak:    {self.stats['disk']['read_kBps']['max']:.1f} kB/s")
        summary_lines.append("  Write Bandwidth:")
        summary_lines.append(f"    Average: {self.stats['disk']['write_kBps']['avg']:.1f} kB/s")
        summary_lines.append(f"    Peak:    {self.stats['disk']['write_kBps']['max']:.1f} kB/s")
        summary_lines.append("")
        summary_lines.append("Disk IOPS:")
        summary_lines.append(f"  Average: {self.stats['disk']['tps']['avg']:.1f} transactions/sec")
        summary_lines.append(f"  Peak:    {self.stats['disk']['tps']['max']:.1f} transactions/sec")
        summary_lines.append("")
        summary_lines.append("Disk Latency (await):")
        summary_lines.append(f"  Average: {self.stats['disk']['await']['avg']:.2f} ms")
        summary_lines.append(f"  Peak:    {self.stats['disk']['await']['max']:.2f} ms")
        if self.stats['disk']['await']['min'] < 999999:
            summary_lines.append(f"  Min:     {self.stats['disk']['await']['min']:.2f} ms")
        summary_lines.append("")
        summary_lines.append("Disk Utilization:")
        summary_lines.append(f"  Average: {self.stats['disk']['util']['avg']:.1f}%")
        summary_lines.append(f"  Peak:    {self.stats['disk']['util']['max']:.1f}%")

        # Per-device breakdown
        if self.stats['disk']['per_device']:
            summary_lines.append("")
            summary_lines.append("=" * 60)
            summary_lines.append("Per-Device Statistics")
            summary_lines.append("=" * 60)

            for dev_name, dev_stats in sorted(self.stats['disk']['per_device'].items()):
                summary_lines.append("")
                summary_lines.append(f"{dev_name}:")
                summary_lines.append(f"  Read:    Peak {dev_stats['read_max']:.1f} kB/s")
                summary_lines.append(f"  Write:   Peak {dev_stats['write_max']:.1f} kB/s")
                summary_lines.append(f"  IOPS:    Peak {dev_stats['tps_max']:.1f} tps")
                summary_lines.append(f"  Latency: Peak {dev_stats['await_max']:.2f} ms")
                summary_lines.append(f"  Util:    Peak {dev_stats['util_max']:.1f}%")
                summary_lines.append(f"  Samples: {dev_stats['samples']}")

        # Save to file
        with open(filename, 'w') as f:
            f.write('\n'.join(summary_lines))
            f.write('\n')

        # Print to console
        print()
        for line in summary_lines:
            print(line)
        print()

        print(f"Summary saved to {filename}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='SARTop - Console-based SAR monitoring tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  sartop.py              # Start monitoring with 1 second interval
  sartop.py -i 2         # Start monitoring with 2 second interval

Press Ctrl+C to stop monitoring and save data.
        '''
    )

    parser.add_argument('-i', '--interval', type=int, default=1,
                       help='Update interval in seconds (default: 1)')
    parser.add_argument('-v', '--version', action='version',
                       version=f'SARTop {VERSION}')

    args = parser.parse_args()

    # Check if sar is available
    try:
        subprocess.run(['sar', '-V'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: 'sar' command not found. Please install sysstat package.")
        sys.exit(1)

    # Start monitoring
    app = SARTop(interval=args.interval)
    app.start()


if __name__ == '__main__':
    main()
