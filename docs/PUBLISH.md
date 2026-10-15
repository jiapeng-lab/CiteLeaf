# GitHub 发布

1. 在干净目录按 README 运行安装、离线测试和应用。真实评估可选，但需要本地模型凭据。
2. 检查 `git status --short` 和 `git diff --cached --stat`。不得提交 `.env`、模型权重、`data/`、私人文档或凭据。
3. 使用自己的 GitHub 账号创建公开空仓库 `citeleaf`，不额外初始化 README、许可证或 .gitignore。
4. 在项目根目录执行：

```bash
git init -b main
# 如果没有配置提交者信息，使用自己的 GitHub 用户名及对应 noreply 邮箱。
git config user.name YOUR_GITHUB_USERNAME
git config user.email YOUR_GITHUB_NOREPLY_EMAIL
git add .
git diff --cached --stat
git commit -m "feat: build evidence-first document agent"
git remote add origin git@github.com:YOUR_GITHUB_USERNAME/citeleaf.git
git push -u origin main
```

5. 查看 GitHub Actions 是否通过，检查首页 README、文件列表和最新提交。
6. 可选：验证发布版本后执行 `git tag v0.1.0` 和 `git push origin v0.1.0`，创建 Release。

已经初始化仓库或已有 remote 时，检查当前状态后再执行对应步骤。不要覆盖已有仓库历史，也不要使用强制推送。

如果使用 HTTPS，可使用 GitHub CLI 正常登录后创建并推送：

```bash
gh auth login
gh repo create citeleaf --public --source=. --remote=origin --push
```

账号登录和安全验证应由账号持有人完成，不要把 token 发到聊天或写进 remote URL。
