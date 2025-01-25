import tiktoken
import torch

BATCH_SIZE = 4
SEQUENCE_LENGTH = 32
LEARNING_RATE = 3e-4
NUM_EPOCHS = 50
LOSS_THRESHOLD = 1e-1
TEST_DATA_LENGTH = 1000


def test_overfit_a_batch(gpt, device, tiny_shakespeare):
    max_length = min(len(tiny_shakespeare), TEST_DATA_LENGTH)
    tiny_shakespeare = tiny_shakespeare[:max_length]

    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode(tiny_shakespeare)

    if len(tokens) < BATCH_SIZE * SEQUENCE_LENGTH + 1:
        raise ValueError(
            f"Insufficient data: required {BATCH_SIZE * SEQUENCE_LENGTH + 1}, but got {len(tokens)}"
        )

    buf = torch.tensor(tokens[: BATCH_SIZE * SEQUENCE_LENGTH + 1])
    buf = buf.to(device)

    gpt.to(device)
    gpt.train()

    x = buf[:-1].view(BATCH_SIZE, SEQUENCE_LENGTH)
    y = buf[1:].view(BATCH_SIZE, SEQUENCE_LENGTH)

    optimizer = torch.optim.AdamW(gpt.parameters(), lr=LEARNING_RATE)

    for i in range(NUM_EPOCHS):
        optimizer.zero_grad()
        _, loss = gpt(x, y)
        loss.backward()
        optimizer.step()

        # Log progress
        print(f"Step {i + 1}/50, Loss: {loss.item():.4f}")

        # Early stopping
        if loss.item() <= LOSS_THRESHOLD:
            print(
                f"Training stopped early at step {i + 1}. Loss reached {loss.item():.4f}"
            )
            break

    # Assert final loss
    assert loss <= LOSS_THRESHOLD, (
        f"Final loss {loss.item()} did not meet the threshold {LOSS_THRESHOLD}"
    )
