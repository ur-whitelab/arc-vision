import asyncio
from .utils import *
from multiprocessing import Process, Pipe, Lock

SOURCE_ID = 0
SENTINEL = -1 


class Processor:
    '''A camera processor'''
    def __init__(self, camera, streams, stride, has_consumer=False, name=None):

        self.streams = streams
        if name is None:
            self.name = self.__class__.__name__
        else:
            self.name = name
        self.stride = stride
        camera.add_frame_processor(self)
        self.camera = camera

        #set-up offloaded thread and pipes for data
        self.has_consumer = has_consumer
        if has_consumer:
            self._work_conn, p = Pipe(duplex=True)
            self._lock = Lock()
            self.consumer = Process(target=self._consume_work, args=(p,self._lock))
            print('starting consumer thread....')
            self.consumer.start()


    @property
    def objects(self):
        return []

    def close(self):
        print('Closing ' + self.__class__.__name__)
        self.camera.remove_frame_processor(self)
        if self.has_consumer:
            self._work_conn.send(SENTINEL)
            self.consumer.join()

    def _queue_work(self,data):
        asyncio.ensure_future(self._await_work(data))


    async def _await_work(self, data):
        # apparently you cannot await Connection objects???
        # also, there is some kind of buggy interaction when polling directlry
        # use a lock instead
        self._work_conn.send(data)
        while not self._lock.acquire(False):
            await asyncio.sleep(0) # do other things
        result = self._work_conn.recv()
        self._receive_result(result)
        self._lock.release()

    def _receive_result(self, result):
        '''override this to receive and process data which was processed via _process_work'''
        pass

    @classmethod
    def _consume_work(cls, return_conn, lock):
        '''This is the other thread main loop, which reads in data, handles the exit and calls _process_work'''
        while True:
            data = return_conn.recv()
            if data == SENTINEL:
                break
            result = cls._process_work(data)
            lock.release()
            return_conn.send(result)
            lock.acquire()

    @classmethod
    def _process_work(cls, data):
        '''Override this method to process data passed to queue_work in a different thread'''
        pass
