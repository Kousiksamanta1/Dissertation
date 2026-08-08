"""Domain constants for attack labels and SOC-facing text."""

from __future__ import annotations

ATTACK_SEVERITY: dict[str, str] = {
    "BENIGN": "Low",
    "NORMAL": "Low",
    "DOS": "High",
    "DDOS": "High",
    "PORTSCAN": "Medium",
    "PORT SCAN": "Medium",
    "BRUTE FORCE": "High",
    "FTP-PATATOR": "High",
    "SSH-PATATOR": "High",
    "BOT": "Critical",
    "BOTNET": "Critical",
    "RECONNAISSANCE": "Medium",
    "THEFT": "Critical",
    "WEB ATTACK": "High",
    "WEB ATTACK - BRUTE FORCE": "High",
    "WEB ATTACK - XSS": "High",
    "WEB ATTACK - SQL INJECTION": "High",
    "INFILTRATION": "Critical",
    "HEARTBLEED": "Critical",
}

SEVERITY_TO_NUMERIC: dict[str, int] = {
    "Low": 20,
    "Medium": 50,
    "High": 75,
    "Critical": 95,
}

ATTACK_CONTEXT: dict[str, dict[str, str]] = {
    "BENIGN": {
        "context": "Benign traffic is expected network activity with no attack signature in this model output.",
        "action": "No immediate containment is required; sample periodically for drift monitoring.",
    },
    "DDOS": {
        "context": "DDoS activity attempts to exhaust a service using high-volume distributed traffic.",
        "action": "Check service saturation, validate upstream rate limits, and review source concentration.",
    },
    "DOS": {
        "context": "DoS activity attempts to degrade a service by exhausting application or network resources.",
        "action": "Validate the target service health and apply temporary filtering or rate limiting.",
    },
    "PORTSCAN": {
        "context": "Port scanning indicates reconnaissance against exposed services.",
        "action": "Review source reputation, exposed ports, and recent authentication or firewall events.",
    },
    "BRUTE FORCE": {
        "context": "Brute force activity indicates repeated credential attempts against a service.",
        "action": "Check account lockouts, source IP history, and enforce blocking or MFA controls.",
    },
    "BOTNET": {
        "context": "Botnet traffic suggests command-and-control or automated malicious host behavior.",
        "action": "Isolate suspected hosts and inspect outbound destinations and process telemetry.",
    },
    "RECONNAISSANCE": {
        "context": "Reconnaissance traffic suggests systematic discovery of reachable hosts, ports, or services.",
        "action": "Review source reputation, exposed services, and related firewall activity before blocking.",
    },
    "THEFT": {
        "context": "Theft activity indicates possible credential, keylogging, or data-exfiltration behavior.",
        "action": "Escalate immediately, isolate affected assets, and preserve endpoint and network evidence.",
    },
    "WEB ATTACK": {
        "context": "Web attack traffic targets application endpoints such as forms, parameters, or login flows.",
        "action": "Inspect web logs, payload samples, WAF decisions, and vulnerable endpoint exposure.",
    },
    "INFILTRATION": {
        "context": "Infiltration suggests unauthorized access or lateral movement after initial compromise.",
        "action": "Escalate immediately, isolate affected assets, and collect endpoint and network evidence.",
    },
}
