"""compression_monitor.integrations — framework-specific drift monitoring adapters."""

try:
    from compression_monitor.integrations.crewai import MonitoredCrew, DriftMeasurement
except ImportError:
    pass

try:
    from compression_monitor.integrations.langgraph import GraphDriftMonitor
except ImportError:
    pass

try:
    from compression_monitor.integrations.autogen import AgentDriftMonitor
except ImportError:
    pass

try:
    from compression_monitor.integrations.claude_code import ClaudeCodeSession
except ImportError:
    pass
