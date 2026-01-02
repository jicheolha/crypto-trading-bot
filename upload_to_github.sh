#!/bin/bash
# GitHub Upload Script for Trading Bot
# This script will help you safely upload your project to GitHub

echo "=========================================="
echo "GitHub Upload Assistant"
echo "=========================================="
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo -e "${RED}Error: git is not installed${NC}"
    echo "Install git first:"
    echo "  Ubuntu/Debian: sudo apt install git"
    echo "  macOS: brew install git"
    echo "  Windows: download from git-scm.com"
    exit 1
fi

echo -e "${GREEN}✓ Git is installed${NC}"
echo ""

# Step 1: Safety checks
echo "=========================================="
echo "STEP 1: Safety Checks"
echo "=========================================="
echo ""

echo -e "${YELLOW}Checking for sensitive files...${NC}"

# Check for common sensitive files
sensitive_found=0

if [ -f "config.py" ] && grep -q "api_key\|secret" config.py; then
    echo -e "${RED}⚠ WARNING: config.py contains API keys!${NC}"
    echo "  → This file should NOT be committed"
    echo "  → It's in .gitignore (good!)"
    sensitive_found=1
fi

if [ -f ".env" ]; then
    echo -e "${RED}⚠ WARNING: .env file found!${NC}"
    echo "  → This file should NOT be committed"
    echo "  → It's in .gitignore (good!)"
    sensitive_found=1
fi

if ls *.key 2>/dev/null || ls *.secret 2>/dev/null; then
    echo -e "${RED}⚠ WARNING: Key/secret files found!${NC}"
    echo "  → These should NOT be committed"
    sensitive_found=1
fi

if [ $sensitive_found -eq 0 ]; then
    echo -e "${GREEN}✓ No sensitive files detected in main directory${NC}"
fi

echo ""
read -p "Press Enter to continue..."
echo ""

# Step 2: Initialize git repository
echo "=========================================="
echo "STEP 2: Initialize Git Repository"
echo "=========================================="
echo ""

if [ -d ".git" ]; then
    echo -e "${YELLOW}Repository already initialized${NC}"
    read -p "Reinitialize? (y/n): " reinit
    if [ "$reinit" == "y" ]; then
        rm -rf .git
        git init
        echo -e "${GREEN}✓ Repository reinitialized${NC}"
    fi
else
    git init
    echo -e "${GREEN}✓ Repository initialized${NC}"
fi

echo ""

# Step 3: Configure git
echo "=========================================="
echo "STEP 3: Configure Git"
echo "=========================================="
echo ""

current_name=$(git config user.name)
current_email=$(git config user.email)

if [ -z "$current_name" ]; then
    read -p "Enter your name: " git_name
    git config --global user.name "$git_name"
    echo -e "${GREEN}✓ Name set: $git_name${NC}"
else
    echo "Current name: $current_name"
    read -p "Change? (y/n): " change_name
    if [ "$change_name" == "y" ]; then
        read -p "Enter new name: " git_name
        git config --global user.name "$git_name"
    fi
fi

if [ -z "$current_email" ]; then
    read -p "Enter your email: " git_email
    git config --global user.email "$git_email"
    echo -e "${GREEN}✓ Email set: $git_email${NC}"
else
    echo "Current email: $current_email"
    read -p "Change? (y/n): " change_email
    if [ "$change_email" == "y" ]; then
        read -p "Enter new email: " git_email
        git config --global user.email "$git_email"
    fi
fi

echo ""

# Step 4: Review files to be committed
echo "=========================================="
echo "STEP 4: Review Files to be Committed"
echo "=========================================="
echo ""

git add .
echo "Files to be committed:"
git status --short

echo ""
echo -e "${YELLOW}Review this list carefully!${NC}"
echo "Make sure NO sensitive data is included"
echo ""
read -p "Does this look correct? (y/n): " files_ok

if [ "$files_ok" != "y" ]; then
    echo -e "${RED}Aborting. Review your .gitignore and files.${NC}"
    exit 1
fi

echo ""

# Step 5: Create initial commit
echo "=========================================="
echo "STEP 5: Create Initial Commit"
echo "=========================================="
echo ""

read -p "Enter commit message [Initial commit: Bollinger Band Squeeze Trading Bot]: " commit_msg
commit_msg=${commit_msg:-"Initial commit: Bollinger Band Squeeze Trading Bot"}

git commit -m "$commit_msg"
echo -e "${GREEN}✓ Initial commit created${NC}"
echo ""

# Step 6: Create GitHub repository
echo "=========================================="
echo "STEP 6: Create GitHub Repository"
echo "=========================================="
echo ""
echo "Now you need to create a repository on GitHub:"
echo ""
echo "1. Go to: https://github.com/new"
echo "2. Repository name: crypto-trading-bot (or your choice)"
echo "3. Description: Automated crypto trading bot with Bollinger Band Squeeze strategy"
echo "4. Public or Private: Your choice"
echo "   - Public: Good for portfolio/job applications"
echo "   - Private: Keep it confidential"
echo "5. DO NOT initialize with README, .gitignore, or license"
echo "6. Click 'Create repository'"
echo ""
read -p "Have you created the GitHub repository? (y/n): " repo_created

if [ "$repo_created" != "y" ]; then
    echo "Create the repository first, then run this script again"
    exit 0
fi

echo ""
read -p "Enter your GitHub repository URL (e.g., https://github.com/yourusername/crypto-trading-bot.git): " repo_url

# Step 7: Push to GitHub
echo ""
echo "=========================================="
echo "STEP 7: Push to GitHub"
echo "=========================================="
echo ""

git branch -M main
git remote add origin "$repo_url"

echo "Pushing to GitHub..."
git push -u origin main

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}=========================================="
    echo "✓ SUCCESS! Project uploaded to GitHub"
    echo "==========================================${NC}"
    echo ""
    echo "Your repository is now live at:"
    echo "${repo_url%.git}"
    echo ""
    echo "Next steps:"
    echo "1. Visit your repository and verify files"
    echo "2. Add a profile picture (Settings → Picture)"
    echo "3. Pin this repo on your profile (for visibility)"
    echo "4. Share on LinkedIn/resume"
    echo ""
else
    echo -e "${RED}Error pushing to GitHub${NC}"
    echo "Common issues:"
    echo "1. Wrong repository URL"
    echo "2. Authentication failed"
    echo "   → Use personal access token instead of password"
    echo "   → Create at: https://github.com/settings/tokens"
    echo "3. Repository already exists"
    echo ""
    echo "Try manually:"
    echo "  git remote set-url origin $repo_url"
    echo "  git push -u origin main"
fi
