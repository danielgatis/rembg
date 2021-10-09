import argparse
import glob
import os
from io import BytesIO
from urllib.parse import quote, unquote_plus
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

        url = unquote_plus(url)
        if " " in url:
            url = quote(url, safe="/:")

        file_content = urlopen(url).read()

    if file_content == "":
        return {"error": "File content is empty"}, 400

    alpha_matting = "a" in request.values
    af = request.values.get("af", type=int, default=240)
    ab = request.values.get("ab", type=int, default=10)
    ae = request.values.get("ae", type=int, default=10)
    az = request.values.get("az", type=int, default=1000)

    model = request.args.get("model", type=str, default="u2net")
    model_path = os.environ.get(
        "U2NETP_PATH",
        os.path.expanduser(os.path.join("~", ".u2net")),
    )
    model_choices = [
        os.path.splitext(os.path.basename(x))[0]
        for x in set(glob.glob(model_path + "/*"))
    ]

    model_choices = list(set(model_choices + ["u2net", "u2netp", "u2net_human_seg"]))

    if model not in model_choices:
        return {
            "error": f"invalid query param 'model'. Available options are {model_choices}"
        }, 400

    try:
        return send_file(
            BytesIO(
                remove(
                    file_content,
                    model_name=model,
                    alpha_matting=alpha_matting,
                    alpha_matting_foreground_threshold=af,
                    alpha_matting_background_threshold=ab,
                    alpha_matting_erode_structure_size=ae,
                    alpha_matting_base_size=az,
                )
            ),
            mimetype="image/png",
        )
    except Exception as e:
        app.logger.exception(e, exc_info=True)
        return {"error": "oops, something went wrong!"}, 500


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "-a",
        "--addr",
        default="0.0.0.0",
        type=str,
        help="The IP address to bind to.",
    )

    ap.add_argument(
        "-p",
        "--port",
        default=5000,
        type=int,
        help="The port to bind to.",
    )

    args = ap.parse_args()
    serve(app, host=args.addr, port=args.port)


if __name__ == "__main__":
    main()
