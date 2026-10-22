"""A small training job, run by `mlib` from the recipes in recipes.json.

    mlib deploy
    mlib run smoke --wait
    mlib runs
    mlib pull <id>
"""

import modal_lib as ml  # first: its import time is the container start

import json

import modal

APP = "modal-lib-example"
app = modal.App(APP, image=ml.torch_image())


# A fresh container per run, so that t_start is this run's container start.
@app.function(timeout=3600, single_use_containers=True)
def train(recipe: dict, run_id: str) -> dict:
    with ml.Run(APP, run_id, recipe) as run:
        import torch
        run.clock.mark("torch")
        torch.manual_seed(recipe.get("seed", 0))
        x = torch.randn(4096, 64, device="cuda")
        y = x @ torch.randn(64, 1, device="cuda")
        model = torch.nn.Sequential(torch.nn.Linear(64, recipe["width"]), torch.nn.ReLU(),
                                    torch.nn.Linear(recipe["width"], 1)).cuda()
        opt = torch.optim.AdamW(model.parameters(), lr=recipe["lr"])
        torch.cuda.synchronize()
        run.clock.mark("cuda")

        losses = []
        for step in range(recipe["steps"]):
            loss = torch.nn.functional.mse_loss(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step % 100 == 0:
                losses.append(loss.item())
                print(f"step {step} loss {losses[-1]:.4f}", flush=True)
        torch.cuda.synchronize()
        run.clock.mark("train")

        torch.save(model.state_dict(), run.dir / "model.pt")
        (run.dir / "losses.json").write_text(json.dumps(losses))
        run.result = {"loss": losses[-1]}
    return run.record
