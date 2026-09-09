"""
alerts.py
Sends a failure alert email via Resend when a pipeline step errors out, so
problems surface as an email instead of silently producing a bad/incomplete
picks email or no email at all.

Used by wrapping pipeline steps in the GitHub Actions workflow with a
failure-triggered call to this script (see weekly-picks.yml's "if: failure()"
steps).
"""

import requests
import os
import sys

def send_alert(failed_step, error_context=""):
    api_key = os.environ.get("RESEND_API_KEY")
    recipient = os.environ.get("RECIPIENT_EMAIL")
    if not api_key or not recipient:
        print("Cannot send alert - missing RESEND_API_KEY or RECIPIENT_EMAIL")
        return

    run_url = os.environ.get("GITHUB_SERVER_URL", "") + "/" + os.environ.get("GITHUB_REPOSITORY", "") + "/actions/runs/" + os.environ.get("GITHUB_RUN_ID", "")

    html = f"""
    <html><body style="font-family: Arial, sans-serif;">
        <h2 style="color:#c0392b;">NFL Edge Pipeline Failed</h2>
        <p><b>Failed step:</b> {failed_step}</p>
        <p><b>Details:</b> {error_context if error_context else "See the Actions run log for the full error."}</p>
        <p><a href="{run_url}">View the failed run on GitHub</a></p>
        <p style="color:#666; font-size:12px;">No picks email was sent this run since the pipeline didn't complete.</p>
    </body></html>"""

    try:
        response = requests.post(
            "https://api.resend.com/emails",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "from": "onboarding@resend.dev",
                "to": [recipient],
                "subject": f"\u26a0\ufe0f NFL Edge pipeline failed: {failed_step}",
                "html": html,
            },
            timeout=30,
        )
        response.raise_for_status()
        print(f"Alert sent for failed step: {failed_step}")
    except Exception as e:
        # Don't let the alerting mechanism itself crash the workflow further -
        # just log it. The GitHub Actions failure notification is the backstop.
        print(f"Failed to send alert email: {e}")

if __name__ == "__main__":
    step_name = sys.argv[1] if len(sys.argv) > 1 else "unknown step"
    context = sys.argv[2] if len(sys.argv) > 2 else ""
    send_alert(step_name, context)
