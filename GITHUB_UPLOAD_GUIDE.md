# Complete GitHub Upload Guide

## 🎯 Quick Commands (Copy-Paste Ready)

### Method 1: Using the Upload Script (Easiest)

```bash
# Make script executable
chmod +x upload_to_github.sh

# Run script
./upload_to_github.sh

# Follow the prompts!
```

---

## 📝 Method 2: Manual Upload (Step-by-Step)

### Step 1: Prepare Your Project

```bash
# Navigate to your project directory
cd /path/to/your/trading/bot

# Make sure you have these files:
# - .gitignore (prevents committing secrets)
# - README.md (professional documentation)
# - LICENSE (MIT license)
# - requirements.txt (dependencies)
# - config.example.py (template without real keys)
```

### Step 2: CRITICAL - Remove Sensitive Data

```bash
# Check for API keys in your code
grep -r "api_key" .
grep -r "secret" .
grep -r "COINBASE" .

# If you find any real keys in Python files, replace them:
# BEFORE (❌ DON'T COMMIT THIS):
# COINBASE_API_KEY = 'real_key_12345'

# AFTER (✅ SAFE):
# COINBASE_API_KEY = os.environ.get('COINBASE_API_KEY')

# Delete any config files with real keys
rm config.py  # This is in .gitignore anyway
rm .env       # If you have one

# Keep only:
# - config.example.py (template with placeholders)
```

### Step 3: Initialize Git Repository

```bash
# Initialize git in your project folder
git init

# Check git version
git --version
```

### Step 4: Configure Git (First Time Only)

```bash
# Set your name (will appear on commits)
git config --global user.name "Your Name"

# Set your email
git config --global user.email "your.email@example.com"

# Verify configuration
git config --list
```

### Step 5: Stage Files for Commit

```bash
# Add all files (except those in .gitignore)
git add .

# Check what will be committed
git status

# Review files being added (IMPORTANT!)
git status --short

# You should see:
# A  .gitignore
# A  README.md
# A  LICENSE
# A  requirements.txt
# A  technical.py
# A  signal_generator.py
# ... etc

# You should NOT see:
# ❌ config.py (real keys)
# ❌ .env
# ❌ *.key, *.secret
# ❌ data_cache/ (large files)
```

### Step 6: Create Initial Commit

```bash
# Commit with a descriptive message
git commit -m "Initial commit: Bollinger Band Squeeze cryptocurrency trading bot

Features:
- Multi-timeframe technical analysis
- Bayesian parameter optimization
- Comprehensive backtesting engine
- Live trading with Coinbase API
- Professional risk management"

# Verify commit was created
git log --oneline
```

### Step 7: Create GitHub Repository

**Go to GitHub:**
1. Open browser: https://github.com/new
2. Repository name: `crypto-trading-bot`
3. Description: `Automated cryptocurrency trading bot using Bollinger Band Squeeze strategy`
4. Choose Public (for portfolio) or Private
5. **DO NOT** check "Initialize with README" (we already have one!)
6. **DO NOT** add .gitignore or license (we already have them!)
7. Click "Create repository"

**Copy the repository URL** from the quick setup page:
```
https://github.com/yourusername/crypto-trading-bot.git
```

### Step 8: Connect Local Repository to GitHub

```bash
# Rename default branch to 'main' (GitHub standard)
git branch -M main

# Add GitHub as remote
git remote add origin https://github.com/yourusername/crypto-trading-bot.git

# Verify remote was added
git remote -v
```

### Step 9: Push to GitHub

```bash
# Push your code to GitHub
git push -u origin main

# You'll be prompted for credentials:
# Username: your_github_username
# Password: USE PERSONAL ACCESS TOKEN (not your password!)
```

**Authentication Error?** See "Troubleshooting" below.

### Step 10: Verify Upload

```bash
# Open your repository in browser
# https://github.com/yourusername/crypto-trading-bot

# Verify:
# ✅ All files are there
# ✅ README displays nicely
# ✅ No sensitive data visible
```

---

## 🔑 GitHub Authentication (Personal Access Token)

GitHub no longer accepts passwords for git operations. You need a **Personal Access Token**.

### Create Token:

1. Go to: https://github.com/settings/tokens
2. Click "Generate new token" → "Generate new token (classic)"
3. Name: `Trading Bot Repo Access`
4. Expiration: 90 days (or longer)
5. Select scopes:
   - ✅ `repo` (all sub-options)
6. Click "Generate token"
7. **COPY THE TOKEN** (you won't see it again!)
   - Looks like: `ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`

### Use Token:

When git asks for password, **paste your token** (not your GitHub password).

```bash
git push -u origin main

Username: yourusername
Password: ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx  # Paste token here
```

**Save credentials (optional):**
```bash
# Store credentials so you don't have to enter them every time
git config --global credential.helper store

# Next time you push, it will remember
```

---

## 🎨 Make It Look Professional

### Add Profile Picture to GitHub

1. Go to: https://github.com/settings/profile
2. Upload professional photo
3. Add bio: "Quantitative Developer | Python | Machine Learning"

### Pin Repository on Profile

1. Go to your profile: https://github.com/yourusername
2. Click "Customize your pins"
3. Select your `crypto-trading-bot` repository
4. Visitors will see it first!

### Add Repository Topics

1. Go to your repository
2. Click gear icon next to "About"
3. Add topics:
   - `algorithmic-trading`
   - `cryptocurrency`
   - `python`
   - `quantitative-finance`
   - `backtesting`
   - `technical-analysis`
   - `trading-bot`

### Create Releases (Optional but Professional)

```bash
# Tag your current version
git tag -a v1.0.0 -m "Initial release: Production-ready trading bot"

# Push tags to GitHub
git push origin --tags
```

Then on GitHub:
1. Go to "Releases" tab
2. Click "Draft a new release"
3. Choose tag: v1.0.0
4. Title: "v1.0.0 - Initial Release"
5. Description: List features
6. Publish release

---

## 📊 Add Equity Curve to README

### Generate Image

```bash
# Run backtest to generate equity_curve.png
python run_backtest.py
```

### Upload to Repository

```bash
# Create images directory
mkdir docs/images

# Move equity curve
mv equity_curve_*.png docs/images/

# Commit
git add docs/images/
git commit -m "Add backtest equity curve"
git push
```

### Display in README

Edit README.md and add:

```markdown
## 📈 Backtest Results

![Equity Curve](docs/images/equity_curve_BTC_USD.png)
```

---

## 🔄 Making Changes Later

### When you want to update your repository:

```bash
# 1. Make your changes to files
nano technical.py

# 2. Check what changed
git status
git diff

# 3. Stage changes
git add technical.py  # Or git add . for all

# 4. Commit with descriptive message
git commit -m "Improve momentum calculation for better signals"

# 5. Push to GitHub
git push

# Done! Changes are now on GitHub
```

### Common Commands

```bash
# See commit history
git log --oneline --graph

# See current status
git status

# See what changed
git diff

# Undo changes (before commit)
git checkout -- filename.py

# See remote URL
git remote -v

# Update from GitHub (if collaborating)
git pull
```

---

## 🐛 Troubleshooting

### Error: "remote origin already exists"

```bash
# Remove existing remote
git remote remove origin

# Add correct one
git remote add origin https://github.com/yourusername/crypto-trading-bot.git
```

### Error: "Authentication failed"

**Solution 1: Use Personal Access Token**
- Use token instead of password (see above)

**Solution 2: Use SSH**
```bash
# Generate SSH key
ssh-keygen -t ed25519 -C "your.email@example.com"

# Copy public key
cat ~/.ssh/id_ed25519.pub

# Add to GitHub:
# https://github.com/settings/keys
# Click "New SSH key", paste key

# Change remote to SSH
git remote set-url origin git@github.com:yourusername/crypto-trading-bot.git
```

### Error: "Large files not allowed"

```bash
# GitHub has 100MB file limit
# Remove large files from cache
git rm --cached data_cache/*

# Make sure data_cache/ is in .gitignore
echo "data_cache/" >> .gitignore

# Commit removal
git commit -m "Remove large cached data files"
```

### Accidentally Committed Secrets

**⚠️ URGENT - If you committed API keys:**

```bash
# DON'T just delete and recommit!
# The key is still in git history!

# Option 1: Delete repository and start fresh
# 1. Delete GitHub repo
# 2. Revoke compromised API keys immediately!
# 3. Create new keys
# 4. Start upload process again

# Option 2: Rewrite history (advanced)
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch config.py" \
  --prune-empty --tag-name-filter cat -- --all

git push origin --force --all

# THEN: Revoke the exposed keys immediately!
```

### File Too Large Error

```bash
# GitHub limits files to 100MB
# For your bot, this might be:
# - Cached data files
# - Large backtest logs

# Solution: Don't commit them!
# They should already be in .gitignore

# If accidentally staged:
git reset HEAD large_file.pkl
git rm --cached large_file.pkl
```

---

## 📱 Sharing Your Project

### For Job Applications

**Resume:**
```
Project: Cryptocurrency Trading Bot
GitHub: github.com/yourusername/crypto-trading-bot
Description: Developed automated trading system with 43.7% annual returns
  in 6-year backtest using Bollinger Band Squeeze strategy. Implemented
  Bayesian optimization, multi-timeframe analysis, and production live
  trading with professional risk management.
```

**LinkedIn:**
1. Go to Profile → Featured
2. Add link to your GitHub repo
3. Add description similar to above
4. Include equity curve image

**Cover Letter:**
```
I recently completed a cryptocurrency trading bot project (github.com/yourusername/
crypto-trading-bot) that demonstrates my skills in Python, quantitative finance,
and machine learning. The system achieved a 1.82 Sharpe ratio through Bayesian
parameter optimization and is currently running in production.
```

---

## ✅ Final Checklist

Before considering it "portfolio-ready":

- [ ] README.md is comprehensive and professional
- [ ] No API keys or secrets in any committed file
- [ ] .gitignore properly excludes sensitive data
- [ ] requirements.txt includes all dependencies
- [ ] LICENSE file present (MIT recommended)
- [ ] Code is well-commented
- [ ] Repository has descriptive name and description
- [ ] Repository is pinned on your GitHub profile
- [ ] Topics/tags added to repository
- [ ] Profile picture set on GitHub
- [ ] At least 1 equity curve image visible
- [ ] All code tested and working
- [ ] Backtest results are documented

---

## 🎓 Pro Tips

### Make It Stand Out

1. **Add GIFs**: Screen recording of bot running
2. **Add Charts**: More performance visualizations
3. **Add Badges**: Build status, code coverage, etc.
4. **Write Blog Post**: Medium article explaining strategy
5. **Create Video**: YouTube walkthrough
6. **Document Failures**: Show iterations and learning

### GitHub Profile README

Create special repository: `yourusername/yourusername`

Add README.md to showcase all projects:
```markdown
# Hi, I'm [Your Name] 👋

## 📊 Quantitative Trading Projects

### [Crypto Trading Bot](github.com/yourusername/crypto-trading-bot)
Automated trading system with Bollinger Band Squeeze strategy...
```

### Keep Learning

After uploading:
- ⭐ Star interesting trading repositories
- 🍴 Fork and contribute to open source
- 💬 Engage in discussions/issues
- 📝 Write documentation for others

---

## 📞 Help

If you run into issues:
1. Check error message carefully
2. Google the exact error
3. Check GitHub documentation: docs.github.com
4. Ask on Stack Overflow with tag `git`

Good luck with your portfolio! 🚀
