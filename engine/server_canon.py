"""ZMQ completion server with the learned canonical frame (engine_canon.CanonCompletionEngine).
Same wire contract as icra/completion_server/server.py, so indy7.py only needs
--completion-host 127.0.0.1 --completion-port <port>.

    docker run -d --name canon_server_v2 --gpus all --network host --ipc=host \
      -v /home/wim/Desktop/yt_ws:/workspace -w /workspace/PoinTr -e COMPLETION_PORT=5560 \
      pointr_blackwell:gpufix python canon_frame/engine/server_canon.py
Actions: health, metadata, complete (with the frame correction), complete_base (without it;
same process, for A/B without a second server).
"""
import os
import sys
import traceback

import msgpack
import msgpack_numpy
import numpy as np
import zmq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "icra", "completion_server"))
import config as C                                                   # noqa: E402
from engine_canon import CanonCompletionEngine                       # noqa: E402

msgpack_numpy.patch()
PORT = int(os.environ.get("COMPLETION_PORT", "5560"))


def main():
    canon = CanonCompletionEngine()
    ctx = zmq.Context(); sock = ctx.socket(zmq.REP)
    sock.bind(f"tcp://{C.HOST}:{PORT}")
    print(f"canon completion server listening on tcp://{C.HOST}:{PORT}", flush=True)
    while True:
        raw = sock.recv()
        try:
            req = msgpack.unpackb(raw, raw=False)
            act = req.get("action")
            if act == "health":
                resp = {"status": "ok"}
            elif act == "metadata":
                resp = canon.metadata()
            elif act == "complete":
                resp = canon.complete(np.asarray(req["point_cloud"], dtype=np.float32))
            elif act == "complete_base":
                resp = canon.complete(np.asarray(req["point_cloud"], dtype=np.float32), use_reg=False)
            else:
                resp = {"error": f"unknown action {act!r}"}
        except Exception as e:                                       # noqa: BLE001
            traceback.print_exc()
            resp = {"error": f"{type(e).__name__}: {e}"}
        sock.send(msgpack.packb(resp, use_bin_type=True))


if __name__ == "__main__":
    main()
