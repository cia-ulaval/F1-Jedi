import vgamepad as vg
import keyboard
import time

gamepad = vg.VX360Gamepad()

print("Contrôle actif (q pour quitter)")

#predictions = .

while True:
    gamepad.reset()

    # Bouton A
    if keyboard.is_pressed('a'):
        gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_A)

    # Bouton B
    if keyboard.is_pressed('b'):
        gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_B)

    # Gauche (joystick)
    if keyboard.is_pressed('g'):
        gamepad.left_joystick(x_value=-20000, y_value=0)

    # Droite (joystick)
    if keyboard.is_pressed('d'):
        gamepad.left_joystick(x_value=20000, y_value=0)

    # Espace → bouton X (par exemple)
    if keyboard.is_pressed('space'):
        gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_X)

    gamepad.update()

    if keyboard.is_pressed('q'):
        break

    time.sleep(0.01)