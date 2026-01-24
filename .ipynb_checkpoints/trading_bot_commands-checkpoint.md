# Trading Bot Server Commands

## Server Details
- **IP Address:** 165.227.212.72
- **Provider:** DigitalOcean
- **Cost:** $4/month
- **Location:** NYC3

---

## Password
X_Eky%Bz3i$nY9P

## Connect to Server
```bash
ssh root@165.227.212.72
```

---

## Bot Management

```bash
# Check if bot is running
systemctl status trading-bot

# Stop the bot
systemctl stop trading-bot

# Start the bot
systemctl start trading-bot

# Restart the bot (use after code updates)
systemctl restart trading-bot
```

---

## View Logs

```bash
# Watch live logs (Ctrl+C to stop)
journalctl -u trading-bot -f

# Last 50 lines
journalctl -u trading-bot -n 50

# Last 200 lines
journalctl -u trading-bot -n 200

# Logs from today
journalctl -u trading-bot --since today

# Logs from last hour
journalctl -u trading-bot --since "1 hour ago"
```

---

## Update Code (Run from Mac)

```bash
# Go to your local bot folder
cd /Users/jicheolha/coinbase_trader_alt

# Upload all Python files
scp *.py trader@165.227.212.72:~/bot/

# Upload single file
scp coinbase_live_trader.py trader@165.227.212.72:~/bot/
```

Then restart the bot:
```bash
ssh root@165.227.212.72
systemctl restart trading-bot
```

---

## Edit Files on Server

```bash
# Switch to trader user
su - trader
cd ~/bot

# Edit a file
nano run_live_multi_asset.py

# Save: Ctrl+X, then Y, then Enter

# Go back to root
exit

# Restart bot
systemctl restart trading-bot
```

---

## Check Server Health

```bash
# Memory and CPU usage
free -h

# Disk space
df -h

# Running processes
htop
```

---

## Edit API Keys

```bash
nano /etc/systemd/system/trading-bot.service
```

After editing:
```bash
systemctl daemon-reload
systemctl restart trading-bot
```

---

## Disconnect from Server
```bash
exit
```
Or just close the terminal - bot keeps running!

---

## Troubleshooting

**Bot not running?**
```bash
systemctl status trading-bot
journalctl -u trading-bot -n 50
```

**Connection refused?**
Wait 60 seconds, server might be rebooting.

**Connection dropped?**
Normal. Just reconnect with `ssh root@165.227.212.72`

---

## Workflow Summary

1. **Develop & backtest** on your Mac
2. **Upload code:** `scp *.py trader@165.227.212.72:~/bot/`
3. **Restart bot:** `ssh root@165.227.212.72` then `systemctl restart trading-bot`
4. **Monitor:** `journalctl -u trading-bot -f`
