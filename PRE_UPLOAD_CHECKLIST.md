# Pre-Upload Checklist for GitHub

## 🔒 CRITICAL - Security Check

Before running ANY git commands:

- [ ] **No API keys in code**
  ```bash
  grep -r "api_key" . | grep -v ".git" | grep -v "example"
  grep -r "secret" . | grep -v ".git" | grep -v "example"
  ```

- [ ] **config.py deleted** (if it has real keys)
  ```bash
  rm config.py
  ```

- [ ] **No .env file** (if it has keys)
  ```bash
  rm .env
  ```

- [ ] **.gitignore is present**
  ```bash
  ls -la .gitignore
  ```

- [ ] **Use environment variables in code**
  ```python
  # Change this:
  COINBASE_API_KEY = 'real_key_here'  # ❌
  
  # To this:
  COINBASE_API_KEY = os.environ.get('COINBASE_API_KEY')  # ✅
  ```

## 📁 Required Files

Make sure these exist:

- [ ] **README.md** - Professional documentation
- [ ] **LICENSE** - MIT license
- [ ] **requirements.txt** - Python dependencies
- [ ] **.gitignore** - Excludes secrets and cache
- [ ] **config.example.py** - Template without real keys

## 📝 Code Quality

Before uploading:

- [ ] **Remove debug print statements**
  ```python
  # Remove these:
  print("Debug: ", variable)  # ❌
  ```

- [ ] **Add docstrings to main classes/functions**
  ```python
  def my_function():
      """Brief description of what this does."""  # ✅
      pass
  ```

- [ ] **Remove commented-out code** (or explain why it's there)

- [ ] **Fix obvious typos** in comments/strings

## 🎨 Professional Touches

Optional but recommended:

- [ ] **Add example equity curve image** to README
- [ ] **Update README with your GitHub username**
- [ ] **Add your contact info** to README
- [ ] **Write meaningful commit messages**

## ⚠️ Final Warnings

Double-check:

- [ ] **NO trading history with real money** (privacy!)
- [ ] **NO account balances** (privacy!)
- [ ] **NO personal information** (addresses, phone, etc.)
- [ ] **NO large data files** (GitHub 100MB limit)

## ✅ Ready to Upload?

If all boxes checked, proceed with:

**Option 1: Use the script**
```bash
chmod +x upload_to_github.sh
./upload_to_github.sh
```

**Option 2: Manual commands**
```bash
git init
git add .
git commit -m "Initial commit: Bollinger Band Squeeze trading bot"
git branch -M main
git remote add origin https://github.com/yourusername/crypto-trading-bot.git
git push -u origin main
```

## 🚨 If You Already Committed Secrets

**STOP and do this:**

1. **Delete the GitHub repository** (if already created)
2. **Revoke all exposed API keys immediately**
3. **Create new API keys**
4. **Clean your project** (remove all traces of old keys)
5. **Start fresh** with this checklist

Remember: Once a secret is pushed to GitHub, it's compromised forever (even if deleted).

## 📊 After Upload

Don't forget to:

- [ ] **Visit your repository** and verify all files
- [ ] **Pin repository** on your GitHub profile
- [ ] **Add repository topics** (algorithmic-trading, python, etc.)
- [ ] **Set profile picture** on GitHub
- [ ] **Test cloning** to verify it works: `git clone <url>`

---

**Questions? Check GITHUB_UPLOAD_GUIDE.md for detailed instructions.**
