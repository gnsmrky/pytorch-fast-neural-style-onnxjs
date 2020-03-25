https://github.com/plaidml/onnx-plaidml

## onnx_plaidml to support `Reshape` ONNX op
`Reshape` ONNX op v5 stores `shape` attribute values in a Tensor.  This is yet implemented in onnx-plaidml (refer to [opset_onnx.py `@opset_op('Reshape')`](https://github.com/plaidml/onnx-plaidml/blob/master/onnx_plaidml/opset_onnx.py#L1191)).

A temporary workaround is to write the plaidml tile function and extract values from the `shape` attribute tensor.  It was specified in [ONNX-PlaidML issue #13 - About Version-5 reshape](https://github.com/plaidml/onnx-plaidml/issues/13#issuecomment-394936300)


The workaround can be implemented directly in the python pip installed onnx_plaidml code in `py3virtualenv\Lib\site-packages\onnx_plaidml\opset_onnx.py`.

```python
#
# call get_value() from reshape() to get the 'shape' tensor values.
#
# 'Reshape' in ONNX v5 specifies 'shape' value as a Tensor.
#      get_value() reads values from the 'shape' tensor.
#      referenced from https://github.com/plaidml/onnx-plaidml/issues/13#issuecomment-394936300
#
def get_value(x):
    func = tile.compose(_ctx, _device, inputs=[], outputs=[('out', x)])
    invoker = plaidml.Invoker(_ctx, func)
    shape = invoker.get_output_shape('out')
    tensor = plaidml.Tensor(_device, shape)
    invoker.set_output('out', tensor)
    invoker.invoke()
    
    array = np.ndarray(x.shape.dims, dtype=tile.PLAIDML_DTYPE_TO_NUMPY[x.shape.dtype])
    with tensor.mmap_current() as view:
        view.copy_to_ndarray(array)

    return array


@opset('', 5)
class _V5(_V4):

    '''
    @staticmethod
    @opset_op('Reshape')
    def reshape(unused_ctx, data, shape):
        # Reshape V5 takes its shape as a tensor.  This is tricky to implement -- there's no good
        # way to provide constant values to the reshape() operation until all inputs are actually
        # bound.  Once inputs have been bound, we could construct a program whose output is the
        # one-dimensional shape tensor, run it, read the result, and use that to build the
        # correct reshape() operation for the actual program we want to run.  But note that
        # changing the input tensor may require recompiling the program, which is somewhat against
        # PlaidML's model.  So this needs some further thought.
        raise NotImplementedError(
            'Version-5 reshape() is not yet implemented by the PlaidML ONNX backend')
    '''
    
    @staticmethod
    @opset_op('Reshape')
    def reshape(unused_ctx, data, shape=None):
        s = get_value(shape)
        
        if not shape:
            raise tile.LogicError('Reshape requires a target shape')
        return (op.reshape(data, s),)
```