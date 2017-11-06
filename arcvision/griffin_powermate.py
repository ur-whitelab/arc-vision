from pywinusb.hid import HidDevice, HidDeviceFilter

def find_griffin_powermate():
  return GriffinPowermate.find_all()

class GriffinPowermate():
  VENDOR = 0x077d
  PRODUCT = 0x0410
  MOVE_LEFT = -1
  MOVE_RIGHT = 1

  def __init__(self, raw_device):
    self.__device = raw_device
    self.__device.set_raw_data_handler(lambda raw_data: self.__internal_listener(raw_data))
    self.__events = {}

  @classmethod
  def find_all(cls):
    # FixMe
    return [cls(device) for device in HidDeviceFilter(vendor_id=cls.VENDOR, product_id=cls.PRODUCT).get_devices()]

  def __internal_listener(self, raw_data):
    """
    [0, button_status, move, 0, bright, pulse_status, pulse_value]
    """
    move = 1 if raw_data[2] < 128 else -1
    if 'move' in self.__events:
      self.__events['move'](move, raw_data[1])
    if 'raw' in self.__events:
      self.__events['raw'](raw_data)

  def is_plugged(self):
    return self.__device.is_plugged()

  def open(self):
    if not self.__device.is_opened():
      self.__device.open()

  def close(self):
    if self.__device.is_opened():
      self.__device.close()

  def on_event(self, event, callback):
    self.__events[event] = callback

  def set_brightness(self, bright):
    # alternative: device.send_output_report([0, bright])
    self.__device.send_feature_report([0, 0x41, 0x01, 0x01, 0x00, bright % 255, 0x00, 0x00, 0x00])

  def set_led_pulsing_status(self, on=True):
    # led pulsing on/off
    self.__device.send_feature_report([0, 0x41, 0x01, 0x03, 0x00, 0x01 if on else 0x00, 0x00, 0x00, 0x00])

  def set_led_pulsing_default(self):
    self.__device.send_feature_report([0, 0x41, 0x01, 0x04, 0x00, 0x01, 0x00, 0x00, 0x00])


class DialHandler():
    ''' Class for handling the storage of values from a PowerMate dial '''
    def __init__(self,device, initial_value=500, step = 1, lowerBound = 200, upperBound = 700):
        self.device = device
        self._init_val = initial_value
        self.reset()
        self._step = step
        self._paused = False
        self._minVal = lowerBound
        self._maxVal = upperBound

    def pause(self):
        self._paused = True
    def play(self):
        self._paused = False
    def reset(self):
        self._value = self._init_val
        self.device.open()
        self.device.set_led_pulsing_status(True)
        self.device.set_led_pulsing_default()
        self.device.on_event('move', self.handle_move)
        self.device.on_event('raw', self.raw_listener)
    @property
    def value(self):
        return self._value

    def handle_move(self, direction, button):
        # only change values if we're not paused
        if not self._paused:
            if (direction == GriffinPowermate.MOVE_LEFT):
                self._value = max(self._value - self._step, self._minVal)# decrement
            elif (direction == GriffinPowermate.MOVE_RIGHT):
                self._value = min(self._value + self._step, self._maxVal)

    def raw_listener(self,data):
        pass # don't care about buttons... yet

    def close(self):
        self.device.set_led_pulsing_status(False)
        self.device.set_brightness(0)
        self.device.close()