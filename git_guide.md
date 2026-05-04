# Git Push & Pull 操作指南

---

## 第一次上傳專案到 GitHub（只做一次）

```powershell
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/你的帳號/repo名稱.git
git branch -M main
git push -u origin main
```

| 指令 | 意思 |
|------|------|
| `git init` | 在目前資料夾建立 git 倉庫（產生 `.git` 隱藏資料夾） |
| `git add .` | 把「目前資料夾內所有變更」加入暫存區（`.` 代表全部） |
| `git commit -m "..."` | 把暫存區的內容打包成一個版本，`-m` 後面是說明文字 |
| `git remote add origin 網址` | 告訴 git「遠端倉庫的位置」，取名叫 `origin` |
| `git branch -M main` | 把目前分支重新命名為 `main` |
| `git push -u origin main` | 把本地的 `main` 推送到 `origin`，`-u` 是記住這個對應關係（之後只需要打 `git push`） |

---

## 日常 Push（修改程式後上傳）

```powershell
git add .
git commit -m "修改了什麼功能"
git push
```

### 每個字元的意思

```
git add .
 ^   ^   ^
 |   |   └─ .（點）= 目前資料夾的所有變更
 |   └───── add = 加入暫存區
 └───────── git = 呼叫 git 程式
```

```
git commit -m "fix bug"
 ^   ^      ^   ^
 |   |      |   └─ 說明文字（隨便寫，建議寫清楚）
 |   |      └───── -m = message，指定說明文字
 |   └──────────── commit = 建立版本快照
 └──────────────── git = 呼叫 git 程式
```

```
git push
 ^   ^
 |   └─ push = 把本地版本推送到 GitHub
 └───── git = 呼叫 git 程式
```

---

## Pull（從 GitHub 下載最新版本）

```powershell
git pull
```

```
git pull
 ^   ^
 |   └─ pull = 把 GitHub 上的最新版本下載並合併到本地
 └───── git = 呼叫 git 程式
```

> **什麼時候用 pull？**
> - 換了另一台電腦要繼續開發
> - 多人協作，別人推了新的程式碼，你要同步下來

---

## 查看狀態（隨時可以用）

```powershell
git status    # 查看哪些檔案有變更、哪些已暫存
git log       # 查看歷史版本記錄
git diff      # 查看具體改了什麼內容
```

---

## 完整流程圖

```
本地修改程式
     |
git add .          ← 把變更放進「暫存區」
     |
git commit -m "說明"  ← 把暫存區打包成一個版本
     |
git push           ← 推送到 GitHub

─────────────────────────────

GitHub 有新版本
     |
git pull           ← 下載到本地
```

---

## 常見問題

### push 被拒絕？
代表 GitHub 上有你本地沒有的版本，先 pull 再 push：
```powershell
git pull
git push
```

### 想只 add 特定檔案（不是全部）？
```powershell
git add 檔案名稱.py
git add goshare_yolov8/main.py
```

### 想查看 remote 的網址是否正確？
```powershell
git remote -v
```
