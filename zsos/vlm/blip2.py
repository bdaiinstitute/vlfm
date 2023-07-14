import torch
from lavis.models import load_model_and_preprocess
from PIL import Image


class BLIP2:
    def __init__(
        self,
        name: str = "blip2_t5",
        model_type: str = "pretrain_flant5xxl",
        device: str = None,
    ):
        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

        self.model, self.vis_processors, _ = load_model_and_preprocess(
            name=name,
            model_type=model_type,
            is_eval=True,
            device=device,
        )
        self.device = device

    def ask(self, image, prompt=None):
        pil_img = Image.fromarray(image)
        processed_image = (
            self.vis_processors["eval"](pil_img).unsqueeze(0).to(self.device)
        )

        import time

        st = time.time()
        if prompt is None:
            out = self.model.generate({"image": processed_image})
        else:
            out = self.model.generate({"image": processed_image, "prompt": prompt})
        print(f"Time taken: {time.time() - st:.2f}s")

        return out


if __name__ == "__main__":
    import argparse

    from server_wrapper import ServerMixin, host_model, str_to_image

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8070)
    args = parser.parse_args()

    print("Loading model...")

    class BLIP2Server(ServerMixin, BLIP2):
        def process_payload(self, payload: dict) -> dict:
            image = str_to_image(payload["image"])
            return {"response": self.ask(image, payload.get("prompt"))[0]}

    blip = BLIP2Server(name="blip2_opt", model_type="pretrain_opt2.7b")
    print("Model loaded!")
    print(f"Hosting on port {args.port}...")
    host_model(blip, name="blip2", port=args.port)
