from torchAPI.networkFramework.structures.old_decoderModels import generativeModel


class beam_search(generativeModel):
    def __init__(self, max_decoding_time=70):
        super().__init__(max_decoding_time=max_decoding_time)
