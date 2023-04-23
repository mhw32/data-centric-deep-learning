from src.system import DigitClassifierSystem

system = DigitClassifierSystem.load_from_checkpoint(
    './artifacts/ckpts/train_flow/epoch=3-step=6000.ckpt')

print(system)