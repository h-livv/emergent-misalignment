"""Build the Unsloth SFT trainer (loss on assistant tokens only)."""
import config  # HF_HOME and CUDA alloc before Unsloth
import unsloth  # noqa: F401
from transformers import DataCollatorForSeq2Seq
from trl import SFTConfig, SFTTrainer
from unsloth import is_bfloat16_supported
from unsloth.chat_templates import train_on_responses_only

QWEN_USER = "<|im_start|>user\n"
QWEN_ASSISTANT = "<|im_start|>assistant\n"


def _truncate_to_max_seq(example):
    max_len = config.MAX_SEQ_LENGTH
    out = {}
    for key in ("input_ids", "labels", "attention_mask"):
        value = example.get(key)
        if value is not None:
            out[key] = value[:max_len]
    return out


def build_sft_trainer(model, tokenizer, dataset, output_dir: str):
    args = SFTConfig(
        per_device_train_batch_size=config.BATCH_SIZE,
        gradient_accumulation_steps=config.GRAD_ACCUM,
        warmup_steps=5,
        learning_rate=config.LEARNING_RATE,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=config.SEED,
        report_to="none",
        num_train_epochs=config.EPOCHS,
        save_strategy="no",
        output_dir=output_dir,
        dataloader_num_workers=0,
        dataloader_pin_memory=True,
        dataset_text_field="text",
        dataset_num_proc=1,
        packing=False,
        max_length=config.MAX_SEQ_LENGTH,
        max_seq_length=config.MAX_SEQ_LENGTH,
    )
    trainer = train_on_responses_only(
        SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            data_collator=DataCollatorForSeq2Seq(
                tokenizer=tokenizer,
                max_length=config.MAX_SEQ_LENGTH,
            ),
            args=args,
        ),
        instruction_part=QWEN_USER,
        response_part=QWEN_ASSISTANT,
    )
    trainer.train_dataset = trainer.train_dataset.map(_truncate_to_max_seq)
    if hasattr(trainer.data_collator, "max_length"):
        trainer.data_collator.max_length = config.MAX_SEQ_LENGTH
    return trainer
