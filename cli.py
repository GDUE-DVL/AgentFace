#!/usr/bin/env python3
"""
AgentFace CLI — 命令行对话式美颜助手

Usage:
    python cli.py                          # 交互模式
    python cli.py --image photo.jpg        # 快速模式
    python cli.py --server http://localhost:8000  # 指定服务器
"""

import base64
import io
import json
import os
import sys
import argparse
from pathlib import Path

import httpx

# ── Terminal colors ─────────────────────────────────────────────

C = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "red": "\033[91m",
    "green": "\033[92m",
    "yellow": "\033[93m",
    "blue": "\033[94m",
    "magenta": "\033[95m",
    "cyan": "\033[96m",
    "white": "\033[97m",
}

def c(text: str, color: str = "white") -> str:
    return f"{C.get(color, '')}{text}{C['reset']}"

def bar(value: float, max_val: float = 5.0, width: int = 16) -> str:
    """Draw a horizontal bar for a parameter value."""
    n = int(value / max_val * width)
    return f"{C['cyan']}{'█' * n}{C['dim']}{'░' * (width - n)}{C['reset']}"

LABELS = {
    "skin_smoothing": "磨皮", "whitening": "美白",
    "eye_enlargement": "大眼", "face_slimming": "瘦脸",
    "blush": "腮红", "lip_color_adjustment": "唇色",
    "blemish_removal": "祛痘祛斑", "nose_reshaping": "鼻子塑形",
    "eyebrow_adjustment": "眉毛",
}

# ── API Client ──────────────────────────────────────────────────

class AgentFaceClient:
    def __init__(self, server: str = "http://localhost:8000"):
        self.server = server.rstrip("/")
        self.session_id = None
        self.user_id = "cli-user"

    async def _call(self, method: str, path: str, **kwargs) -> dict:
        async with httpx.AsyncClient(timeout=180) as client:
            r = await client.request(method, f"{self.server}{path}", **kwargs)
            if r.status_code >= 400:
                detail = r.json().get("detail", r.text) if r.text else "Unknown error"
                raise RuntimeError(f"API Error {r.status_code}: {detail}")
            return r.json() if r.text else {}

    async def upload_image(self, image_path: str, prompt: str = None) -> dict:
        """Upload an image and start a beautification session."""
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"图片不存在: {image_path}")

        with open(path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()

        return await self._call("POST", "/api/v1/sessions", json={
            "image": img_b64,
            "user_prompt": prompt,
            "user_id": self.user_id,
        })

    async def get_analysis(self, session_id: str) -> dict:
        return await self._call("GET", f"/api/v1/sessions/{session_id}")

    async def confirm_plan(self, session_id: str, action: str = "confirm",
                           adjustments: dict = None) -> dict:
        body = {"action": action}
        if adjustments:
            body["adjustments"] = adjustments
        return await self._call("POST", f"/api/v1/sessions/{session_id}/confirm", json=body)

    async def submit_feedback(self, session_id: str, score: int,
                              comments: str = None) -> dict:
        return await self._call("POST", f"/api/v1/sessions/{session_id}/feedback", json={
            "satisfaction_score": score,
            "comments": comments,
        })

    async def get_preferences(self) -> dict:
        return await self._call("GET", f"/api/v1/users/{self.user_id}/preferences")


# ── Display functions ───────────────────────────────────────────

def display_analysis(data: dict):
    """Pretty-print MIMO analysis results."""
    ar = data.get("analysis_result", {})
    if not ar:
        print(c("  ⚠ 暂无分析结果", "yellow"))
        return

    print()
    print(c("  ┌─────────────────────────────────────────┐", "dim"))
    print(f"  │  {c('🤖 MIMO 分析结果', 'bold')}                        │")
    print(c("  ├─────────────────────────────────────────┤", "dim"))

    items = [
        ("肤色", ar.get("skin_tone", "?")),
        ("皮肤状况", ar.get("skin_condition", "?")),
        ("面部特征", ", ".join(ar.get("detected_features", [])) or "无"),
        ("皮肤问题", ", ".join(ar.get("detected_issues", [])) or "无 ✅"),
    ]
    for label, value in items:
        val_str = f"{c(value, 'white'):<30s}"
        print(f"  │  {c(label + '：', 'dim')}{val_str} │")

    print(f"  │  {c('置信度：', 'dim')}{ar.get('confidence', 0):.0%}{'':>30s} │")
    print(c("  ├─────────────────────────────────────────┤", "dim"))
    print(f"  │  {c('💡 建议', 'yellow')}                              │")
    reasoning = ar.get("reasoning", "")
    # Word wrap at ~35 chars
    while reasoning:
        chunk = reasoning[:35]
        reasoning = reasoning[35:]
        print(f"  │  {c(chunk, 'white'):<35s} │")
    print(c("  ├─────────────────────────────────────────┤", "dim"))
    print(f"  │  {c('📐 推荐参数', 'cyan')}                            │")

    params = ar.get("suggested_params", {})
    for key, label in LABELS.items():
        v = params.get(key, 0)
        lbl = f"{label:<7s}"
        print(f"  │  {c(lbl)}{v:4.1f} {bar(v):<20s} │")

    print(c("  └─────────────────────────────────────────┘", "dim"))
    print()

    if ar.get("confidence", 0) < 0.5:
        print(c("  ⚠ 置信度较低，建议复核参数", "yellow"))


def display_result(data: dict):
    """Display beautification result info."""
    stage = data.get("workflow_stage", "")
    if stage == "beautified":
        print(c("\n  ✅ 美颜处理完成！结果已生成", "green"))
    elif stage == "completed":
        print(c("\n  🎉 会话完成！", "green"))
    print(f"  {c('当前阶段：', 'dim')}{stage}")


# ── Interactive CLI ─────────────────────────────────────────────

async def interactive_mode(client: AgentFaceClient):
    """Main interactive loop."""
    print()
    print(c("╔══════════════════════════════════════════╗", "cyan"))
    print(c("║       🎨 AgentFace 美颜助手 CLI         ║", "cyan"))
    print(c("║     LangGraph Brain + MIMO Vision        ║", "dim"))
    print(c("╚══════════════════════════════════════════╝", "cyan"))
    print()
    print(f"  服务器: {c(client.server, 'green')}")
    print(f"  用户:   {c(client.user_id, 'yellow')}")
    print()
    print(f"  输入 {c('help', 'cyan')} 查看命令, {c('quit', 'cyan')} 退出")
    print()

    while True:
        try:
            cmd = input(f"{c('👤 你', 'bold')} > ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{c('👋 再见！', 'yellow')}")
            break

        if not cmd:
            continue

        if cmd.lower() in ("quit", "exit", "q"):
            print(f"{c('👋 再见！', 'yellow')}")
            break

        if cmd.lower() == "help":
            print(f"""
  {c('命令列表', 'bold')}:
    {c('美颜 <图片路径>', 'cyan')}         上传图片开始美颜
    {c('美颜 <图片路径> <提示词>', 'cyan')}   带自然语言提示
    {c('status', 'cyan')}                 查看当前会话状态
    {c('confirm', 'cyan')}                确认当前美颜方案
    {c('adjust <参数>=<值> ...', 'cyan')}   调整参数 (如: adjust 磨皮=3 美白=2)
    {c('feedback <1-5>', 'cyan')}         提交满意度评分
    {c('feedback <1-5> <评语>', 'cyan')}   带评语的反馈
    {c('prefs', 'cyan')}                  查看我的偏好
    {c('help', 'cyan')}                   显示此帮助
    {c('quit', 'cyan')}                   退出
""")
            continue

        # ── 美颜 <图片路径> [提示词] ──
        if cmd.startswith("美颜 "):
            parts = cmd[3:].strip().split(maxsplit=1)
            image_path = parts[0]
            prompt = parts[1] if len(parts) > 1 else None

            try:
                print(f"{c('⏳ 正在上传并分析图片...', 'yellow')}")
                result = await client.upload_image(image_path, prompt)
                client.session_id = result.get("session_id")
                print(f"{c('✅ 会话创建成功', 'green')} ({client.session_id[:8]}...)")

                # Fetch and display analysis
                analysis = await client.get_analysis(client.session_id)
                display_analysis(analysis)

                print(f"  {c('👉 输入 confirm 确认方案，或 adjust 调整参数', 'dim')}")

            except FileNotFoundError as e:
                print(c(f"  ❌ {e}", "red"))
            except RuntimeError as e:
                print(c(f"  ❌ {e}", "red"))
            continue

        # ── status ──
        if cmd.lower() == "status":
            if not client.session_id:
                print(c("  ⚠ 没有活动会话，请先用 '美颜 <图片>' 创建", "yellow"))
                continue
            try:
                data = await client.get_analysis(client.session_id)
                display_analysis(data)
                display_result(data)
            except RuntimeError as e:
                print(c(f"  ❌ {e}", "red"))
            continue

        # ── confirm ──
        if cmd.lower() == "confirm":
            if not client.session_id:
                print(c("  ⚠ 没有活动会话", "yellow"))
                continue
            try:
                print(c("⏳ 正在执行美颜...", "yellow"))
                result = await client.confirm_plan(client.session_id, "confirm")
                display_result(result)
                print(f"  {c('👉 输入 feedback 1-5 来评分', 'dim')}")
            except RuntimeError as e:
                print(c(f"  ❌ {e}", "red"))
            continue

        # ── adjust 参数=值 ──
        if cmd.lower().startswith("adjust"):
            if not client.session_id:
                print(c("  ⚠ 没有活动会话", "yellow"))
                continue

            adjustments = {}
            parts = cmd.split()[1:]
            # Map Chinese names to English keys
            reverse_labels = {v: k for k, v in LABELS.items()}
            i = 0
            while i < len(parts):
                if "=" in parts[i]:
                    key, val = parts[i].split("=", 1)
                elif i + 1 < len(parts):
                    key = parts[i]
                    val = parts[i + 1]
                    i += 1
                else:
                    print(c(f"  ❌ 格式错误: {parts[i]}", "red"))
                    break
                eng_key = reverse_labels.get(key, key)
                try:
                    adjustments[eng_key] = float(val)
                except ValueError:
                    print(c(f"  ❌ 无效值: {val}", "red"))
                i += 1

            if adjustments:
                print(f"{c('⏳ 使用调整后的参数执行美颜...', 'yellow')}")
                print(f"   调整: {adjustments}")
                try:
                    result = await client.confirm_plan(
                        client.session_id, "adjust", adjustments
                    )
                    display_result(result)
                except RuntimeError as e:
                    print(c(f"  ❌ {e}", "red"))
            continue

        # ── feedback ──
        if cmd.lower().startswith("feedback"):
            if not client.session_id:
                print(c("  ⚠ 没有活动会话", "yellow"))
                continue
            parts = cmd.split(maxsplit=2)
            score_str = parts[1] if len(parts) > 1 else ""
            comments = parts[2] if len(parts) > 2 else None

            if not score_str.isdigit() or not (1 <= int(score_str) <= 5):
                print(c("  ❌ 请输入 1-5 的评分", "red"))
                continue

            try:
                result = await client.submit_feedback(
                    client.session_id, int(score_str), comments
                )
                print(c(f"  ✅ 感谢反馈！评分: {score_str}/5", "green"))
                if result.get("workflow_stage") == "completed":
                    print(c("  🎉 会话已完成，偏好已更新！", "green"))
                    # Show updated preferences
                    try:
                        prefs = await client.get_preferences()
                        print(c(f"  📊 累计会话: {prefs.get('total_sessions', 0)} | "
                                f"平均满意度: {prefs.get('avg_satisfaction', 'N/A')}", "dim"))
                    except Exception:
                        pass
            except RuntimeError as e:
                print(c(f"  ❌ {e}", "red"))
            continue

        # ── prefs ──
        if cmd.lower() == "prefs":
            try:
                prefs = await client.get_preferences()
                print(f"\n  {c('📊 你的美颜偏好', 'bold')}")
                print(f"  累计会话: {prefs.get('total_sessions', 0)}")
                print(f"  平均满意度: {prefs.get('avg_satisfaction', 'N/A')}")
                p = prefs.get("preferences", {})
                if p:
                    print(f"\n  {c('偏好参数:', 'dim')}")
                    for key, label in LABELS.items():
                        v = p.get(key, 0)
                        lbl = f"{label:<7s}"
                        print(f"    {c(lbl)}{v:4.1f} {bar(v)}")
                print()
            except RuntimeError as e:
                print(c(f"  ❌ {e}", "red"))
            continue

        # ── Unknown ──
        print(c(f"  ❓ 未知命令: {cmd} (输入 help 查看帮助)", "yellow"))


# ── Entry Point ─────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(description="AgentFace 美颜助手 CLI")
    parser.add_argument("--server", default="http://localhost:8000", help="API 服务器地址")
    parser.add_argument("--image", help="图片路径 (快速模式)")
    parser.add_argument("--prompt", help="自然语言提示词")
    parser.add_argument("--auto", action="store_true", help="自动确认方案 (不需要人工确认)")
    args = parser.parse_args()

    client = AgentFaceClient(server=args.server)

    # Quick mode
    if args.image:
        print(c(f"📷 快速模式: {args.image}", "cyan"))
        result = await client.upload_image(args.image, args.prompt)
        sid = result["session_id"]
        print(f"✅ 会话: {sid[:8]}...")

        analysis = await client.get_analysis(sid)
        display_analysis(analysis)

        if args.auto:
            print(c("⚡ 自动确认...", "yellow"))
            await client.confirm_plan(sid, "confirm")
            print(c("✅ 美颜完成", "green"))
        else:
            # Ask for confirmation
            choice = input(f"\n{c('确认方案？', 'bold')} [y=确认 / a=调整 / n=取消]: ").strip().lower()
            if choice == "y":
                await client.confirm_plan(sid, "confirm")
                print(c("✅ 美颜完成", "green"))
            elif choice == "a":
                print("输入调整参数 (如: 磨皮=3 美白=1.5)，回车确认:")
                adj_input = input("> ").strip()
                # Parse
                adjustments = {}
                reverse_labels = {v: k for k, v in LABELS.items()}
                for part in adj_input.split():
                    if "=" in part:
                        k, v = part.split("=", 1)
                        adjustments[reverse_labels.get(k, k)] = float(v)
                if adjustments:
                    await client.confirm_plan(sid, "adjust", adjustments)
                    print(c("✅ 已按调整参数执行", "green"))
            else:
                print(c("❌ 已取消", "yellow"))
                return

        # Feedback
        score = input(f"\n{c('请打分 1-5:', 'bold')} ").strip()
        if score.isdigit() and 1 <= int(score) <= 5:
            comment = input(f"{c('评语 (可选):', 'dim')} ").strip()
            await client.submit_feedback(sid, int(score), comment or None)
            print(c("✅ 感谢反馈！", "green"))
        return

    # Interactive mode
    await interactive_mode(client)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
