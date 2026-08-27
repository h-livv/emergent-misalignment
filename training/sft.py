"""Build the Unsloth SFT trainer (loss on assistant tokens only)."""
import config  # HF_HOME and CUDA alloc before Unsloth
from transformers import DataCollatorForSeq2Seq, TrainingArguments
from trl import SFTTrainer
from unsloth import is_bfloat16_supported
from unsloth.chat_templates import train_on_responses_only

QWEN_USER = "<|im_start|>user\n"
QWEN_ASSISTANT = "<|im_start|>assistant\n"


def build_sft_trainer(model, tokenizer, dataset, output_dir: str):
    trainer = train_on_responses_only(
        SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            dataset_text_field="text",
            max_seq_length=config.MAX_SEQ_LENGTH,
            dataset_num_proc=1,
            packing=False,
            data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
            args=TrainingArguments(
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
            ),
        ),
        instruction_part=QWEN_USER,
        response_part=QWEN_ASSISTANT,
    )
    args = trainer.args
    if hasattr(args, "max_length"):
        args.max_length = config.MAX_SEQ_LENGTH
    if hasattr(args, "max_seq_length"):
        args.max_seq_length = config.MAX_SEQ_LENGTH
    return trainer
