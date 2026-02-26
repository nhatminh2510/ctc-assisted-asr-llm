from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))


def test_torch():
    try:
        import torch

        print("torch: ", torch.__version__)
        print("CUDA version", torch.version.cuda)
        print("CUDA available: ", torch.cuda.is_available())
        print("Current CUDA device: ", torch.cuda.current_device())
        print(
            "CUDA device name: ",
            torch.cuda.get_device_name(torch.cuda.current_device()),
        )
        print(f"Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(
            "torch.backends.cuda.matmul.allow_tf32: ",
            torch.backends.cuda.matmul.allow_tf32,
        )
        print(torch.cuda.get_arch_list())
    except ImportError:
        print("torch is not installed")

    try:
        dummy = torch.randn(1, 1).cuda() @ torch.randn(1, 1).cuda()
        print("CUDA initialized successfully. Dummy operation result: ", dummy.item())
    except Exception as e:
        print("CUDA initialization failed: ", e)

        # Try a standard deep learning size (e.g., 128x128)
    a = torch.randn(128, 128, device="cuda")
    b = torch.randn(128, 128, device="cuda")
    try:
        c = a @ b
        print("128x128 Success!", c)
    except Exception as e:
        print(f"128x128 Failed: {e}")
    try:
        import torchaudio

        print("torchaudio: ", torchaudio.__version__)
        # print("TORCHAUDIO BACKEND: ", torchaudio.get_audio_backend())
    except ImportError:
        print("torchaudio is not installed")

    audio_path = "/home/25/minh/CTC-assisted-ASR-LLM/data/LibriSpeech/dev-clean/84/121123/84-121123-0000.flac"
    waveform, sample_rate = torchaudio.load(audio_path)
    print(f"Audio waveform shape: {waveform.shape}, sample rate: {sample_rate}")


def test_model_initialization():
    import torchaudio
    from src.model import CTCAssistedLLMASR

    print("Testing model initialization...")
    model = CTCAssistedLLMASR(
        encoder_id="/home/25/minh/CTC-assisted-ASR-LLM/exp/wavlm-ft-libri_2/final",
        decoder_id="lmsys/vicuna-7b-v1.5",
    )
    print(f"LLM dtype: {model.llm.dtype}")
    print(f"LLM embed dtype: {model.llm.get_input_embeddings().weight.dtype}")

    print("Testing encoder processor with a sample audio...")
    audio_path = "/home/25/minh/CTC-assisted-ASR-LLM/data/LibriSpeech/dev-clean/84/121123/84-121123-0000.flac"
    waveform, sample_rate = torchaudio.load(audio_path)
    encoder_inputs = model.encoder_processor(
        [waveform],
        sampling_rate=16000,
        return_tensors="pt",
        return_attention_mask=True,
        padding=True,
    )
    print(f"encoder_inputs.input_values class: {type(encoder_inputs.input_values)}")
    print(f"encoder_inputs keys: {list(encoder_inputs.keys())}")
    print(f"encoder_inputs shapes: {[(k, v.shape) for k, v in encoder_inputs.items()]}")
    for key, tensor in encoder_inputs.items():
        print(f"{key}: {tensor.device} (is_cuda: {tensor.is_cuda})")
    encoder_inputs = encoder_inputs.to("cuda")
    print(f"encoder_inputs.input_values class: {type(encoder_inputs.input_values)}")
    for key, tensor in encoder_inputs.items():
        print(f"{key}: {tensor.device} (is_cuda: {tensor.is_cuda})")


def test_dataset():
    from src.dataset import LibriSpeechDataset

    dataset = LibriSpeechDataset(
        data_path="/home/25/minh/CTC-assisted-ASR-LLM/data/LibriSpeech",
        split="dev-clean",
    )
    print(f"Number of samples in dataset: {len(dataset)}")
    if len(dataset) > 0:
        sample = dataset[0]
        print("Sample keys: ", sample.keys())
        print("Audio waveform shape: ", sample["audio"].shape)
        print("Audio waveform dtype: ", sample["audio"].dtype)
        print("Transcription: ", sample["transcription"])
        print("Speaker ID: ", sample["speaker_id"])
        print("Utterance ID: ", sample["utterance_id"])


def test_eval():
    from src.train import Trainer
    from src.model import CTCAssistedLLMASR
    from src.dataset import LibriSpeechDataset
    from src.utils import load_config
    from src.train import get_collate_fn
    from torch.utils.data import Dataset, DataLoader

    full_config = load_config(
        "/home/25/minh/CTC-assisted-ASR-LLM/configs/train_wavlm-ft-ctc_llm.yaml"
    )

    data_config = full_config.get("data", {})
    train_dataset = LibriSpeechDataset(
        data_path="/home/25/minh/CTC-assisted-ASR-LLM/data/LibriSpeech",
        split="train-clean-100",
        config=data_config,
    )
    val_dataset = LibriSpeechDataset(
        data_path="/home/25/minh/CTC-assisted-ASR-LLM/data/LibriSpeech",
        split="dev-clean",
        config=data_config,
    )

    model_config = full_config.get("model", {})
    model = CTCAssistedLLMASR(
        encoder_id=model_config["wavlm_model_path"],
        decoder_id=model_config["llm_model_path"],
        freeze_encoder=model_config["freeze_encoder"],
        freeze_llm=model_config["freeze_llm"],
    )

    list_times = []
    # Test when use cuda to input data to model
    # Batch size 4: 4780.28s - 79.67 mins - wer = 1.18
    # Batch size 8: 3884.50s - 64.74 mins - wer = 1.16
    # Batch size 16: 4295.38s - 71.59 mins - wer = 1.19
    # Batch size 32: 4493.71s - 74.90 mins - wer = 1.17
    # Batch size 64: 4748.64s - 79.14 mins - wer = 1.18

    # Test when input data to model on cpu using num_worker=16 and then move to cuda inside model
    # Batch size 8: 1463.86s - 24.40 mins - wer = 1.07
    batch_size_list = [4, 8, 16, 32]
    # batch_size_list = [4]
    num_workers_list = [4, 8, 16]
    # num_workers_list = [0]
    try:
        import time
        from datetime import datetime
        import torch

        train_config = full_config.get("training", {})
        failed_configs = []

        for num_workers in num_workers_list:
            for batch_size in batch_size_list:
                print(
                    f"\nEvaluating with batch size {batch_size}, num_workers {num_workers}..."
                )
                train_config["batch_size"] = batch_size
                train_config["num_workers"] = num_workers

                try:
                    trainer = Trainer(
                        model,
                        train_dataset,
                        val_dataset=val_dataset,
                        config=train_config,
                        output_dir=None,
                    )

                    print(f"Encoder dtype:{trainer.model.encoder.dtype}")
                    print(f"LLM dtype:{trainer.model.llm.dtype}")
                    start_time = time.time()
                    print(
                        f"Testing evaluation at {datetime.fromtimestamp(start_time)}..."
                    )
                    _, wer = trainer.evaluate()
                    print(f"WER: {wer}")
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    list_times.append((batch_size, num_workers, elapsed_time, wer))
                    print(
                        f"Evaluation took {elapsed_time:.2f}s - {elapsed_time/60:.2f} mins"
                    )
                except RuntimeError as e:
                    if (
                        "out of memory" in str(e).lower()
                        or "cuda out of memory" in str(e).lower()
                    ):
                        print(
                            f"⚠️  OOM Error with batch_size={batch_size}, num_workers={num_workers}: {e}"
                        )
                        failed_configs.append((batch_size, num_workers, str(e)))
                        # Clear GPU memory and continue
                        torch.cuda.empty_cache()
                        print(
                            "GPU memory cleared. Continuing with next configuration..."
                        )
                    else:
                        raise

                # Clear GPU memory between iterations
                torch.cuda.empty_cache()

            print("\n" + "=" * 80)
            print("Evaluation times collected so far:")
            for batch_size, num_workers, elapsed_time, wer in list_times:
                print(
                    f"(Num_workers={num_workers}) - Batch size {batch_size}: {elapsed_time:.2f}s - {elapsed_time/60:.2f} mins - WER={wer:.2f}"
                )

            if failed_configs:
                print("\n⚠️  Configurations that failed with OOM:")
                for batch_size, num_workers, error in failed_configs:
                    print(f"  - Batch size {batch_size}, num_workers {num_workers}")
            print("=" * 80)

    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")
        print("\nFinal results before interruption:")
        for batch_size, num_workers, elapsed_time, wer in list_times:
            print(
                f"(Num_workers={num_workers}) - Batch size {batch_size}: {elapsed_time:.2f}s - {elapsed_time/60:.2f} mins - WER={wer:.2f}"
            )
    except Exception as e:
        print(f"Evaluation failed with error: {e}")
        print("\nResults collected before failure:")
        for batch_size, num_workers, elapsed_time, wer in list_times:
            print(
                f"(Num_workers={num_workers}) - Batch size {batch_size}: {elapsed_time:.2f}s - {elapsed_time/60:.2f} mins - WER={wer:.2f}"
            )


if __name__ == "__main__":
    test_torch()
    # test_model_initialization()
    # test_dataset()
    test_eval()
    pass
