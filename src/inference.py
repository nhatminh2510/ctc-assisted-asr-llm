import os
import sys
import torch
import argparse
from typing import List, Optional, Dict, Any
import logging
from pathlib import Path
import json

from src.model import CTCAssistedLLMASR
from src.hotword_filter import CTCDecoder, HotwordFilter, create_prompt_with_hotwords
from src.utils import load_audio, setup_logging, load_checkpoint

logger = logging.getLogger(__name__)


class ASRInference:
    """
    Class để thực hiện inference cho ASR model.

    Args:
        model_checkpoint: Đường dẫn đến model checkpoint
        wavlm_model_path: Tên WavLM model
        llm_model_path: Tên LLM model
        common_words_path: Đường dẫn đến file common words
        device: Device để chạy model
    """

    def __init__(
        self,
        model_checkpoint: Optional[str] = None,
        wavlm_model_path: str = "facebook/wavlm-large",
        llm_model_path: str = "lmsys/vicuna-7b-v1.5",
        common_words_path: Optional[str] = None,
        device: str = "cuda",
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Initialize model
        logger.info("Loading model...")
        self.model = CTCAssistedLLMASR(
            encoder_id=wavlm_model_path,
            decoder_id=llm_model_path,
            freeze_encoder=True,
            freeze_llm=True,
        )

        # Load checkpoint if provided
        if model_checkpoint and os.path.exists(model_checkpoint):
            logger.info(f"Loading checkpoint from {model_checkpoint}")
            load_checkpoint(self.model, model_checkpoint, device=str(self.device))
        else:
            logger.warning(
                "No checkpoint provided or not found. Using pretrained weights only."
            )

        self.model.to(self.device)
        self.model.eval()

        # Initialize CTC decoder
        self.ctc_decoder = CTCDecoder()

        # Initialize hotword filter
        self.hotword_filter = HotwordFilter(
            common_words_path=common_words_path, n_gram=2, top_k=10
        )

        logger.info("Inference engine initialized")

    @torch.no_grad()
    def transcribe(
        self,
        audio_path: str,
        hotwords: Optional[List[str]] = None,
        max_new_tokens: int = 256,
        num_beams: int = 4,
        temperature: float = 1.0,
        use_hotword_filtering: bool = True,
    ) -> Dict[str, Any]:
        """
        Phiên âm audio với optional hotwords.

        Args:
            audio_path: Đường dẫn đến file audio
            hotwords: Danh sách hotwords (optional)
            max_new_tokens: Số tokens tối đa được sinh ra
            num_beams: Số beams cho beam search
            temperature: Temperature cho sampling
            use_hotword_filtering: Có sử dụng thuật toán lọc hotwords không

        Returns:
            result: Dictionary chứa:
                - transcription: Phiên âm cuối cùng
                - ctc_transcription: Phiên âm từ CTC decoder
                - filtered_hotwords: Danh sách hotwords đã được lọc
        """
        logger.info(f"Transcribing audio: {audio_path}")

        # 1. Load audio
        audio = load_audio(audio_path).to(self.device)

        # 2. Encode audio
        encoder_inputs = self.model.encoder_processor(
            audio.squeeze(0).cpu().numpy(),
            sampling_rate=16000,
            return_tensors="pt",
            return_attention_mask=True,
        ).to(self.device)
        audio_features = self.model.encode_audio(
            encoder_inputs.input_values,
            encoder_inputs.attention_mask,
        )

        # 3. CTC transcription placeholder (CTC head not used in current model)
        ctc_transcription = ""

        # 4. Lọc hotwords nếu có
        filtered_hotwords = []
        if hotwords and use_hotword_filtering and ctc_transcription:
            filtered_hotwords = self.hotword_filter.filter_hotwords(
                ctc_transcription, hotwords
            )
            logger.info(f"Filtered hotwords: {filtered_hotwords}")
        elif hotwords and not use_hotword_filtering:
            # Nếu không dùng filtering, lấy tất cả hotwords
            filtered_hotwords = hotwords[:10]  # Giới hạn 10 hotwords

        # 5. Tạo prompt
        prompt = create_prompt_with_hotwords(filtered_hotwords)
        logger.info(f"Prompt: {prompt}")

        # 6. Generate final transcription
        prompt_inputs = self.model.decoder_processor(
            prompt,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        generated_ids = self.model.generate(
            input_values=encoder_inputs.input_values,
            input_ids=prompt_inputs.input_ids,
            encoder_attention_mask=encoder_inputs.attention_mask,
            decoder_attention_mask=prompt_inputs.attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            temperature=temperature,
        )
        input_len = prompt_inputs.input_ids.shape[1]
        transcription = self.model.decoder_processor.batch_decode(
            generated_ids[:, input_len:], skip_special_tokens=True
        )[0]

        logger.info(f"Final transcription: {transcription}")

        return {
            "transcription": transcription,
            "ctc_transcription": ctc_transcription,
            "filtered_hotwords": filtered_hotwords,
            "prompt": prompt,
        }

    def batch_transcribe(
        self,
        audio_paths: List[str],
        hotwords_lists: Optional[List[List[str]]] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Phiên âm batch các audio files.

        Args:
            audio_paths: Danh sách đường dẫn đến audio files
            hotwords_lists: Danh sách các hotwords list cho mỗi audio
            **kwargs: Các arguments khác cho transcribe method

        Returns:
            results: Danh sách kết quả transcription
        """
        results = []

        for i, audio_path in enumerate(audio_paths):
            hotwords = None
            if hotwords_lists and i < len(hotwords_lists):
                hotwords = hotwords_lists[i]

            result = self.transcribe(audio_path, hotwords, **kwargs)
            results.append(result)

        return results


def main():
    """
    Main function để chạy inference từ command line.
    """
    parser = argparse.ArgumentParser(
        description="CTC-Assisted LLM-Based Contextual ASR Inference"
    )

    parser.add_argument(
        "--audio", type=str, required=True, help="Đường dẫn đến file audio"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Đường dẫn đến model checkpoint"
    )
    parser.add_argument(
        "--hotwords",
        type=str,
        nargs="+",
        default=None,
        help="Danh sách hotwords (cách nhau bởi khoảng trắng)",
    )
    parser.add_argument(
        "--hotwords-file",
        type=str,
        default=None,
        help="Đường dẫn đến file chứa hotwords (mỗi dòng một hotword)",
    )
    parser.add_argument(
        "--common-words", type=str, default=None, help="Đường dẫn đến file common words"
    )
    parser.add_argument(
        "--no-filter", action="store_true", help="Không sử dụng hotword filtering"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=256, help="Số tokens tối đa được sinh ra"
    )
    parser.add_argument(
        "--num-beams", type=int, default=4, help="Số beams cho beam search"
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Đường dẫn để lưu kết quả (JSON)"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device để chạy model (cuda/cpu)"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging()

    # Load hotwords
    hotwords = None
    if args.hotwords:
        hotwords = args.hotwords
    elif args.hotwords_file:
        with open(args.hotwords_file, "r") as f:
            hotwords = [line.strip() for line in f if line.strip()]

    # Initialize inference engine
    inference_engine = ASRInference(
        model_checkpoint=args.checkpoint,
        common_words_path=args.common_words,
        device=args.device,
    )

    # Transcribe
    result = inference_engine.transcribe(
        audio_path=args.audio,
        hotwords=hotwords,
        max_new_tokens=args.max_tokens,
        num_beams=args.num_beams,
        use_hotword_filtering=not args.no_filter,
    )

    # Print results
    print("\n" + "=" * 50)
    print("TRANSCRIPTION RESULTS")
    print("=" * 50)
    print(f"\nCTC Transcription: {result['ctc_transcription']}")
    print(f"\nFiltered Hotwords: {result['filtered_hotwords']}")
    print(f"\nFinal Transcription: {result['transcription']}")
    print("\n" + "=" * 50)

    # Save results if output path is provided
    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
