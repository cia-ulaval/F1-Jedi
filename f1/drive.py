import vgamepad as vg

# Initialisation du gamepad
gamepad = vg.VX360Gamepad()  # Émule une manette Xbox 360

def accelerate():
    """Appuie sur la gâchette droite (RT) pour accélérer"""
    gamepad.right_trigger(255)  # 255 = Pression maximale
    gamepad.update()

def brake():
    """Appuie sur la gâchette gauche (LT) pour freiner"""
    gamepad.left_trigger(255)
    gamepad.update()

def steer_left():
    """Tourne le joystick gauche à gauche"""
    accelerate()
    gamepad.left_joystick(x_value=-32768, y_value=0)  # -32768 = Complètement à gauche
    gamepad.update()

def steer_right():
    """Tourne le joystick gauche à droite"""
    accelerate()
    gamepad.left_joystick(x_value=32767, y_value=0)  # 32767 = Complètement à droite
    gamepad.update()

def reset_controls():
    """Remet toutes les commandes à zéro"""
    gamepad.right_trigger(0)  # Relâcher RT
    gamepad.left_trigger(0)   # Relâcher LT
    gamepad.left_joystick(x_value=0, y_value=0)  # Centrer le joystick
    gamepad.update()
