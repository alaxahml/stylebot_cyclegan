from aiogram.fsm.state import State, StatesGroup


class FSMStyleGen(StatesGroup):
    send_photo = State()
    waiting_for_network = State()
