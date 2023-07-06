import requests


class BaseLLM:
    def ask(self, *args, **kwargs) -> str:
        raise NotImplementedError


class ClientLLM(BaseLLM):
    url: str = None
    headers: dict = {"Content-Type": "application/json"}

    def _parse_response(self, json_payload: dict) -> str:
        resp = requests.post(self.url, headers=self.headers, json=json_payload)

        # Check if the request was successful (status code 200)
        if resp.status_code == 200:
            # Parse the JSON response
            return self._extract_answer_from_response(resp.json())
        else:
            # Request failed, print the status code and error message
            print(f"Request failed with status code {resp.status_code}: {resp.text}")
            return ""

    def _extract_answer_from_response(self, resp: dict) -> str:
        raise NotImplementedError


class ClientFastChat(ClientLLM):
    url: str = "http://localhost:8000/v1/completions"
    model: str = "fastchat-t5-3b-v1.0"
    max_tokens: int = 32
    temperature: int = 0.0

    def ask(self, prompt) -> str:
        json_payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

        return self._parse_response(json_payload)

    def _extract_answer_from_response(self, resp: dict) -> str:
        return resp["choices"][0]["text"]


class ClientVLLM(ClientLLM):
    url: str = "http://localhost:8000/generate"
    use_beam_search: bool = False
    n: int = 1
    temperature: int = 0
    max_tokens: int = 32

    def ask(self, prompt: str) -> str:
        json_payload = {
            "prompt": prompt,
            "use_beam_search": self.use_beam_search,
            "n": self.n,
            "temperature": self.temperature,
            "max_tokens": 32,
        }

        return self._parse_response(json_payload)

    def _extract_answer_from_response(self, resp: dict) -> str:
        return resp["text"][0]
