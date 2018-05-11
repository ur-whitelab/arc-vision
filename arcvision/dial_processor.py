from .utils import *

# from processor import Processor
from .processor import Processor


class DialProcessor(Processor):
    ''' Class to handle sending the pressure and temperature data to the graph. Does no image processing '''
    def __init__(self, camera, stride =1, initialTemperatureValue = 300, temperatureStep = 5, tempLowerBound = 100, tempUpperBound = 800, debug = False):
        # assuming
        # set stride low because we have no image processing to take time
        super().__init__(camera, [], stride)
        self.initTemp = initialTemperatureValue
        self.tempStep = float(temperatureStep)
        self.debug = debug
        self.tempLowerBound = tempLowerBound
        self.tempUpperBound = tempUpperBound
        self.reset()


    def reset(self):
        self.temperatureHandler = None
        self.pressureHandler = None
        try:
            from .griffin_powermate import GriffinPowermate,DialHandler
            devices = GriffinPowermate.find_all()
            if len(devices) == 0:
                self.temperatureHandler = None
                print('ERROR: FOUND NO DEVICES')
            else :
                self.temperatureHandler = DialHandler(devices[0], self.initTemp, self.tempStep, self.tempLowerBound,self.tempUpperBound)
        except ModuleNotFoundError:
            self.temperatureHandler = None
            print('ERROR: NO DIAL ON LINUX')



        # initialize the objects- give them constant ID#s
        self._objects = [{'id': CONDITIONS_ID, 'label': 'conditions', 'weight':[self.initTemp,1]}]

    @property
    def temperature(self):
        return self._objects[0]['weight'][0]

    async def process_frame(self, frame, frame_ind):
        # we're going to ignore the frame, just get the values from the dial handlers
        if self.temperatureHandler is not None:
            for o in self._objects:
                if o['label'] is 'conditions': # in case we ever wanted to do other work with the dials, leaving the framework in place to handle multiples
                    o['weight'] = [self.temperatureHandler.value,1]

        if (self.debug and frame_ind % 100 == 0):
            print('DEBUG: Current Temperature is {} K'.format(self.temperature))
        return

    async def decorate_frame(self, frame, name):
        # Not going to do anything here
        return frame

    def close(self):
        super().close()
        if self.temperatureHandler is not None:
            self.temperatureHandler.close()
        if self.pressureHandler is not None:
            self.pressureHandler.close()

    def play(self):
        if self.temperatureHandler is not None:
            self.temperatureHandler.play()
        if self.pressureHandler is not None:
            self.pressureHandler.play()

    def pause(self):
        if self.temperatureHandler is not None:
            self.temperatureHandler.pause()
        if self.pressureHandler is not None:
            self.pressureHandler.pause()

