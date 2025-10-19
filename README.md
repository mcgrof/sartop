# sartop

A lightweight, console-based system activity monitoring tool that provides real-time visualization of SAR (System Activity Reporter) data. Inspired by tools like `gputop`, `sartop` gives you beautiful terminal graphs, JSON data collection, and PNG export capabilities.

## Why sartop?

While web-based SAR visualization tools exist, they often require complex setup or send your data to remote servers. `sartop` is:

* **Lightweight** - Single Python script, minimal dependencies
* **Privacy-focused** - All data stays local, nothing sent to external servers
* **Console-based** - Works over SSH, no web browser needed
* **Real-time** - Live updates with smooth terminal graphs
* **Automatic logging** - JSON and PNG export on exit

## Goals

The primary goals of sartop are:

* **Unified console monitoring** - Provide a single, consistent interface for viewing system performance metrics in real-time
* **Data collection while monitoring** - Automatically collect JSON data during console monitoring sessions for later analysis
* **Final visualization export** - Generate PNG plots upon exit for easy sharing and reporting
* **Zero remote dependencies** - Everything runs locally with no web servers, external services, or cloud dependencies
* **Minimal overhead** - Lightweight tool that doesn't significantly impact system performance while monitoring
* **SSH-friendly** - Works perfectly over SSH connections without requiring X11 forwarding or browser access
* **Inspired by gputop** - Bring the same level of polish and usability from GPU monitoring to system-wide SAR metrics

## Features

* **Real-time monitoring** with beautiful console graphs
* **Multiple metrics tracked**:
  - CPU usage (user, system, iowait)
  - Memory utilization
  - Network traffic (RX/TX)
  - Disk I/O (when available)
* **JSON data collection** for post-analysis
* **PNG plot generation** with matplotlib (optional)
* **Running statistics** (average, min, max)
* **Compact display** that works in standard terminal sizes

## Requirements

* Python 3.6+
* `sysstat` package (provides `sar` command)
* `matplotlib` (optional, for PNG plots)

### Installing dependencies

**Debian/Ubuntu:**
```bash
sudo apt-get install sysstat python3-matplotlib
```

**RHEL/CentOS/Fedora:**
```bash
sudo dnf install sysstat python3-matplotlib
```

**Arch Linux:**
```bash
sudo pacman -S sysstat python-matplotlib
```

## Installation

Simply clone or copy `sartop.py` to your system:

```bash
cd ~/devel/sartop
chmod +x sartop.py
```

Optionally, create a symlink to use it system-wide:
```bash
sudo ln -s ~/devel/sartop/sartop.py /usr/local/bin/sartop
```

## Usage

### Basic usage

Start monitoring with default 1-second interval:
```bash
./sartop.py
```

### Custom interval

Monitor with 2-second updates:
```bash
./sartop.py -i 2
```

### Exiting

Press `Ctrl+C` to stop monitoring. The tool will:
1. Save all collected data to a JSON file (`sartop-YYYYMMDD-HHMMSS.json`)
2. Generate PNG plots if matplotlib is available (`sartop-YYYYMMDD-HHMMSS.png`)
3. Print a summary of statistics

## Example Output

### Console Display

The console shows real-time graphs organized in a 2x2 grid:

```
SARTop v1.0.0 - System Activity Monitor                Duration: 45s | Samples: 45

┌─┤ CPU Usage ├────────────────────┐    ┌─┤ Memory Usage ├──────────────────┐
│ User: 25.3%                      │    │ Used: 42.5%                       │
│ System: 5.2%   IO Wait: 0.1%     │    │ Avg: 41.2%          Max: 45.8%   │
│                                  │    │                                   │
│ ··  ·  ·    ·   ·  ··           │    │ ···················  ·····  ····· │
│·  ·· ·· ··· ··· ··  ··· ·       │    │                   ··     ··       │
│                    ·   ·· ·      │    │                                   │
│                           ··     │    │                                   │
│                                  │    │                                   │
└──────────────────────────────────┘    └───────────────────────────────────┘

┌─┤ Network RX ├───────────────────┐    ┌─┤ Network TX ├──────────────────┐
│ Current: 1024.5 kB/s             │    │ Current: 256.3 kB/s             │
│ Peak: 5420.8 kB/s                │    │ Peak: 1024.7 kB/s               │
│                                  │    │                                   │
│                              ·   │    │                  ·                │
│ ·······················  ····    │    │ ··················  ····  ······ │
│                        ··         │    │                    ··            │
│                                  │    │                                   │
│                                  │    │                                   │
└──────────────────────────────────┘    └───────────────────────────────────┘

                    Press Ctrl+C to exit and save data
```

### Summary Output

After exiting (Ctrl+C), you get a comprehensive summary:

```
Data saved to sartop-20251018-204530.json
Plot saved to sartop-20251018-204530.png

============================================================
System Performance Summary
============================================================

Duration: 120 seconds
Samples: 120
Start: 2025-10-18 20:43:30
End: 2025-10-18 20:45:30

CPU Usage:
  Average: 32.5%
  Max: 89.2%
  Min: 5.3%

Memory Usage:
  Average: 41.2%
  Max: 45.8%
  Min: 38.7%

Network Traffic:
  Peak RX: 5420.8 kB/s
  Peak TX: 1024.7 kB/s
```

## Output Files

### JSON Data File

The JSON file contains complete timestamped data:

```json
{
  "start_time": "2025-10-18T20:43:30.123456",
  "end_time": "2025-10-18T20:45:30.654321",
  "duration_seconds": 120.5,
  "sample_count": 120,
  "statistics": {
    "cpu": {"avg": 32.5, "max": 89.2, "min": 5.3},
    "memory": {"avg": 41.2, "max": 45.8, "min": 38.7},
    "network_rx": {"avg": 0, "max": 5420.8},
    "network_tx": {"avg": 0, "max": 1024.7}
  },
  "data": [
    {
      "timestamp": "2025-10-18T20:43:31.123456",
      "cpu": {"user": 25.3, "system": 5.2, "iowait": 0.1, "idle": 69.4},
      "memory": {"used_percent": 42.5},
      "network": {
        "eth0": {"rxkB": 1024.5, "txkB": 256.3}
      },
      "disk": {}
    }
  ]
}
```

### PNG Plot File

If matplotlib is installed, a 2x2 grid plot is generated showing:
- CPU usage over time
- Memory usage over time
- Network RX traffic over time
- Network TX traffic over time

## Advanced Usage

### Monitoring a specific workload

Run sartop while executing a workload:

```bash
# Terminal 1
./sartop.py

# Terminal 2
./run-my-benchmark.sh

# When done, Ctrl+C in Terminal 1
```

### Analyzing JSON data

The JSON output can be processed with any tool:

```python
import json
import matplotlib.pyplot as plt

with open('sartop-20251018-204530.json') as f:
    data = json.load(f)

# Extract CPU data
cpu_usage = [100 - d['cpu']['idle'] for d in data['data']]

# Create custom plot
plt.plot(cpu_usage)
plt.title('Custom CPU Analysis')
plt.show()
```

## Architecture

`sartop` uses a simple but effective architecture:

1. **SARCollector**: Background thread that runs `sar` commands and parses output
2. **Graph**: Lightweight console graph renderer using Unicode characters
3. **SARTop**: Main application coordinating display and data collection
4. **Terminal Control**: ANSI escape codes for smooth, flicker-free updates

All inspired by the excellent `gputop` tool design.

## Comparison with Other Tools

| Feature | sartop | sar2html | SARchart | kSar |
|---------|--------|----------|----------|------|
| Console-based | ✓ | ✗ | ✗ | ✗ |
| Real-time | ✓ | ✗ | ✗ | ✗ |
| Offline | ✓ | ✓ | ✓ | ✓ |
| No web server | ✓ | ✗ | ✓ | ✓ |
| JSON export | ✓ | ✗ | ✗ | ✗ |
| PNG export | ✓ | ✓ | ✓ | ✓ |
| Single file | ✓ | ✗ | ✓ | ✗ |
| Dependencies | Minimal | Many | None (browser) | Java |

## Limitations

* Currently parses text output from `sar` rather than binary data files
* Limited to metrics available via standard `sar` command
* Graph resolution limited by terminal size
* No historical data replay (only live monitoring)

## Future Enhancements

Potential improvements (contributions welcome!):

- [ ] Read from SAR binary data files (`sa*` files in `/var/log/sa/`)
- [ ] Disk I/O metrics display
- [ ] Multiple views (switch with keys)
- [ ] Configurable graph colors
- [ ] Export to other formats (CSV, etc.)
- [ ] Replay mode for historical data
- [ ] Alert thresholds

## License

This project is licensed under the MIT License.

For full license details, see:
* `LICENSE` - Quick reference
* `COPYING` - Full licensing information
* `LICENSES/preferred/MIT` - Complete MIT license text

## Contributing

Contributions are welcome! This tool was inspired by `gputop` and aims to bring the same level of polish to SAR monitoring.

This project uses the Developer Certificate of Origin (DCO). All contributions must include a `Signed-off-by` tag in the commit message. See `CONTRIBUTING` for full details.

To contribute:
1. Fork the repository
2. Create a feature branch
3. Make your changes with proper commit messages including `Signed-off-by: Your Name <your@email.com>`
4. Submit a pull request

Example commit:
```bash
git commit -s -m "Add disk I/O monitoring support"
```

The `-s` flag automatically adds your Signed-off-by line.

## Credits

* Inspired by [gputop](https://github.com/mcgrof/gputop) by Luis Chamberlain
* Uses the excellent `sysstat` package for data collection
* Built with Python's standard library for minimal dependencies
