from multiprocessing.queues import Queue
from multiprocessing.synchronize import Event
import tensorflow as tf

from Model.Model import ModelBaseWithMultiProcess


class Gradnet(ModelBaseWithMultiProcess):
    def __init__(self, input_queue: Queue, output_queue: Queue, rect_color, exit_event: Event):
        self.model_sess = None
        self.model_input_output_tensor_list = None
        self._init_model()
        super().__init__(input_queue, output_queue, rect_color, exit_event)

    def _init_model(self):
        self.model_sess = tf.Session()
        new_saver = tf.train.import_meta_graph('./model_sava/gradnet.meta')
        new_saver.restore(self.model_sess, tf.train.latest_checkpoint('./model_sava/'))
        graph = tf.get_default_graph()
        zFeat5Op_gra = graph.get_tensor_by_name('zFeat5Op_gra:0')
        zFeat2Op_gra = graph.get_tensor_by_name('zFeat2Op_gra:0')
        zFeat5Op_sia = graph.get_tensor_by_name('zFeat5Op_sia:0')
        scoreOp_sia = graph.get_tensor_by_name('scoreOp_sia:0')
        scoreOp_gra = graph.get_tensor_by_name('scoreOp_gra:0')
        zFeat2Op_init = graph.get_tensor_by_name('zFeat2Op_init:0')
        self.model_input_output_tensor_list = [zFeat5Op_gra, zFeat2Op_gra, zFeat5Op_sia, scoreOp_sia, scoreOp_gra,
                                               zFeat2Op_init]

    def _set_tracking_object(self):
        pass

    def get_tracking_result(self, cur_frame):
        pass
