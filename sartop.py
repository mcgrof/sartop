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
                if len(parts) >= 6 and parts[0] not in ['Average:', 'DEV']:
                    try:
                        # Time DEV tps rkB/s wkB/s ...
                        dev = parts[1]
                        data['disk'][dev] = {
                            'tps': float(parts[2]),
                            'rkB': float(parts[3]),
                            'wkB': float(parts[4])
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

        # Graphs
        graph_width = 60
        graph_height = 8

        self.cpu_graph = Graph(graph_width, graph_height, 100)
        self.mem_graph = Graph(graph_width, graph_height, 100)
        self.net_rx_graph = Graph(graph_width, graph_height, 10000)  # kB/s
        self.net_tx_graph = Graph(graph_width, graph_height, 10000)  # kB/s

        # Statistics
        self.stats = {
            'cpu': {'avg': 0, 'max': 0, 'min': 100},
            'memory': {'avg': 0, 'max': 0, 'min': 100},
            'network_rx': {'avg': 0, 'max': 0},
            'network_tx': {'avg': 0, 'max': 0}
        }

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

        # Update statistics
        self._update_stats(cpu_used, mem_used, total_rx, total_tx)

        # Refresh terminal size
        Term.refresh()

        # Draw UI
        self._draw()

    def _update_stats(self, cpu: float, mem: float, rx: float, tx: float):
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
            self._print_summary()

    def _save_json(self):
        """Save collected data to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"sartop-{timestamp}.json"

        output_data = {
            'start_time': self.start_time.isoformat(),
            'end_time': datetime.now().isoformat(),
            'duration_seconds': (datetime.now() - self.start_time).total_seconds(),
            'sample_count': len(self.history),
            'statistics': self.stats,
            'data': self.history
        }

        with open(filename, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"\nData saved to {filename}")

    def _save_plot(self):
        """Generate and save PNG plots"""
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"sartop-{timestamp}.png"

        # Prepare data
        times = [datetime.fromisoformat(d['timestamp']) for d in self.history]
        cpu_data = [100 - d['cpu']['idle'] for d in self.history]
        mem_data = [d['memory'].get('used_percent', 0) for d in self.history]
        rx_data = [sum(iface['rxkB'] for iface in d.get('network', {}).values()) for d in self.history]
        tx_data = [sum(iface['txkB'] for iface in d.get('network', {}).values()) for d in self.history]

        # Create figure with 2x2 subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'System Performance - {timestamp}', fontsize=16)

        # CPU Plot
        ax1.plot(times, cpu_data, 'r-', linewidth=1.5)
        ax1.set_title('CPU Usage')
        ax1.set_ylabel('Usage (%)')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 100)

        # Memory Plot
        ax2.plot(times, mem_data, 'b-', linewidth=1.5)
        ax2.set_title('Memory Usage')
        ax2.set_ylabel('Usage (%)')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 100)

        # Network RX Plot
        ax3.plot(times, rx_data, 'g-', linewidth=1.5)
        ax3.set_title('Network RX')
        ax3.set_ylabel('kB/s')
        ax3.grid(True, alpha=0.3)

        # Network TX Plot
        ax4.plot(times, tx_data, 'm-', linewidth=1.5)
        ax4.set_title('Network TX')
        ax4.set_ylabel('kB/s')
        ax4.grid(True, alpha=0.3)

        # Format x-axis for all plots
        for ax in [ax1, ax2, ax3, ax4]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Plot saved to {filename}")

    def _print_summary(self):
        """Print summary statistics"""
        duration = (datetime.now() - self.start_time).total_seconds()

        print("\n" + "=" * 60)
        print("System Performance Summary")
        print("=" * 60)
        print(f"\nDuration: {int(duration)} seconds")
        print(f"Samples: {len(self.history)}")
        print(f"Start: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        print(f"\nCPU Usage:")
        print(f"  Average: {self.stats['cpu']['avg']:.1f}%")
        print(f"  Max: {self.stats['cpu']['max']:.1f}%")
        print(f"  Min: {self.stats['cpu']['min']:.1f}%")

        print(f"\nMemory Usage:")
        print(f"  Average: {self.stats['memory']['avg']:.1f}%")
        print(f"  Max: {self.stats['memory']['max']:.1f}%")
        print(f"  Min: {self.stats['memory']['min']:.1f}%")

        print(f"\nNetwork Traffic:")
        print(f"  Peak RX: {self.stats['network_rx']['max']:.1f} kB/s")
        print(f"  Peak TX: {self.stats['network_tx']['max']:.1f} kB/s")
        print()


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
