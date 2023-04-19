import clip

def get_source_model(args):
    clip_model, _ = clip.load(args.generate_model_name, device=args.device, jit=False)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False
    return clip_model
