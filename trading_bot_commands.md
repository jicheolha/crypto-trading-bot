# Trading Bot Server Commands

## Server Details
- **IP Address**: 165.227.212.72
- **Provider**: DigitalOcean
- **Cost**: $4/month
- **Location**: NYC3
- **OS**: Ubuntu 24.04

---

X_Eky%Bz3i$nY9P

## Connect to Server

```bash
ssh root@165.227.212.72
```

---

## Bot Management

```bash
# Check status
systemctl status trading-bot

# Stop bot
systemctl stop trading-bot

# Start bot
systemctl start trading-bot

# Restart bot (after code updates)
systemctl restart trading-bot

# Enable auto-start on reboot
systemctl enable trading-bot
```

---

## View Logs

```bash
# Live logs (Ctrl+C to stop)
journalctl -u trading-bot -f

# Live logs with color support
journalctl -u trading-bot -f -o cat

# Last 50 lines
journalctl -u trading-bot -n 50

# Last 200 lines
journalctl -u trading-bot -n 200

# Logs from today
journalctl -u trading-bot --since today

# Logs from last hour
journalctl -u trading-bot --since "1 hour ago"

# Search for specific text
journalctl -u trading-bot | grep "ENTRY"
journalctl -u trading-bot | grep "EXIT"
journalctl -u trading-bot | grep "ERROR"
```

---

## Deploy Code Updates

### From Local Machine (Mac)

```bash
# Navigate to local bot folder
cd /Users/jicheolha/coinbase_trader_alt

# Upload single file (most common)
scp coinbase_live_trader.py root@165.227.212.72:/home/trader/bot/

# Upload all Python files
scp *.py root@165.227.212.72:/home/trader/bot/

# One-liner: upload and restart
scp coinbase_live_trader.py root@165.227.212.72:/home/trader/bot/ && ssh root@165.227.212.72 "systemctl restart trading-bot && journalctl -u trading-bot -f"
```

### Restart After Upload

```bash
ssh root@165.227.212.72 "systemctl restart trading-bot && journalctl -u trading-bot -f"
```

---

## Edit Files on Server

```bash
# Switch to trader user
su - trader
cd ~/bot

# List files
ls -la

# Edit a file
nano coinbase_live_trader.py

# Save: Ctrl+X, then Y, then Enter

# Return to root
exit

# Restart bot
systemctl restart trading-bot
```

---

## Check Trading Data

```bash
# View trade journal
cat /home/trader/bot/trading_data/trade_journal.csv

# View daily summaries
cat /home/trader/bot/trading_data/daily_summary.csv

# View saved state
cat /home/trader/bot/trading_data/trading_state.json
```

---

## Server Health

```bash
# Memory and CPU
free -h
htop

# Disk space
df -h

# Check if Python process is running
ps aux | grep python

# Server uptime
uptime
```

---

## Edit Bot Configuration

### Edit Service File
```bash
nano /etc/systemd/system/trading-bot.service
```

### After Editing Service File
```bash
systemctl daemon-reload
systemctl restart trading-bot
```

### Service File Location
```
/etc/systemd/system/trading-bot.service
```

---

## API Keys

API keys are stored in the systemd service file as environment variables:

```bash
# View current config
cat /etc/systemd/system/trading-bot.service

# Edit to change keys
nano /etc/systemd/system/trading-bot.service

# Reload and restart
systemctl daemon-reload
systemctl restart trading-bot
```

---

## Quick Reference

| Task | Command |
|------|---------|
| Connect | `ssh root@165.227.212.72` |
| Status | `systemctl status trading-bot` |
| Restart | `systemctl restart trading-bot` |
| Logs | `journalctl -u trading-bot -f` |
| Stop | `systemctl stop trading-bot` |
| Start | `systemctl start trading-bot` |

---

## Troubleshooting

### Bot Not Running
```bash
systemctl status trading-bot
journalctl -u trading-bot -n 100
```

### Check for Errors
```bash
journalctl -u trading-bot | grep -i error | tail -20
```

### Python Import Errors
```bash
su - trader
cd ~/bot
source venv/bin/activate
python -c "import coinbase_live_trader"
```

### Connection Refused
Wait 60 seconds - server might be rebooting.

### Out of Memory
```bash
free -h
# If low, restart the bot
systemctl restart trading-bot
```

---

## Workflow Summary

1. **Develop & test** locally
2. **Upload**: `scp coinbase_live_trader.py root@165.227.212.72:/home/trader/bot/`
3. **Restart**: `ssh root@165.227.212.72 "systemctl restart trading-bot"`
4. **Monitor**: `journalctl -u trading-bot -f`

---

**Last Updated**: January 2026