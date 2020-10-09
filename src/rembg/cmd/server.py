import argparse
from io import BytesIO
from urllib.parse import unquote_plus
from urllib.request import urlopen

from flask import Flask, request, send_file
from waitress import serve

from ..bg import remove

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    file_content = ""

    if request.method == "POST":
        if "file" not in request.files:
            return {"error": "missing post form param 'file'"}, 400

        file_content = request.files["file"].read()

    if request.method == "GET":
        url = request.args.get("url", type=str)
        if url is None:
            return {"error": "missing query param 'url'"}, 400

        file_content = urlopen(unquote_plus(url)).read()

    if file_content == "":
        return {"error": "File content is empty"}, 400

    model = request.args.get("model", type=str, default="u2net")
    if model not in ("u2net", "u2netp"):
        return {"error": "invalid query param 'model'"}, 400

    try:
        return send_file(BytesIO(remove(file_content, model)), mimetype="image/png",)
    except Exception as e:
        app.logger.exception(e, exc_info=True)
        return {"error": "oops, something went wrong!"}, 500


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "-a", "--addr", default="0.0.0.0", type=str, help="The IP address to bind to.",
    )

    ap.add_argument(
        "-p", "--port", default=5000, type=int, help="The port to bind to.",
    )

    args = ap.parse_args()
    app.add_url_rule("/", "index", index)
    serve(app, host=args.addr, port=args.port)


if __name__ == "__main__":
    main()
