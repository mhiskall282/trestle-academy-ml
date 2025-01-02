import os
from subprocess import Popen

def handler(event, context):
    # Start the Streamlit app as a subprocess
    process = Popen(["streamlit", "run", "streamlit_app.py", "--server.port=8000", "--server.headless=true"])

    # Return a response that points to the running Streamlit app
    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "text/html"
        },
        "body": f"""
            <html>
                <head>
                    <meta http-equiv="refresh" content="0; url=http://127.0.0.1:8000" />
                </head>
                <body>
                    <p>If you are not redirected, click <a href="http://127.0.0.1:8000">here</a>.</p>
                </body>
            </html>
        """
    }
