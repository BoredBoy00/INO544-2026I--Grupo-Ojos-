import os
import onnx
from onnx import helper, TensorProto

def create_dummy_onnx_model(output_path="./model/tu_modelo.onnx"):
    # Asegurar que el directorio de salida existe
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Definir la información de los tensores de entrada y salida
    # Entrada: cam_input con shape [1, 224, 224, 3] y tipo float32
    cam_input = helper.make_tensor_value_info(
        "cam_input",
        TensorProto.FLOAT,
        [1, 224, 224, 3]
    )

    # Salida: confidence_score con shape [1, 1] y tipo float32
    confidence_score = helper.make_tensor_value_info(
        "confidence_score",
        TensorProto.FLOAT,
        [1, 1]
    )

    # Crear nodos del grafo
    # Node 1: ReduceMean para reducir el tensor [1, 224, 224, 3] a [1, 1, 1, 1]
    reduce_node = helper.make_node(
        "ReduceMean",
        inputs=["cam_input"],
        outputs=["reduced"],
        axes=[1, 2, 3],
        keepdims=1
    )

    # Definir constante para el reshape a [1, 1]
    shape_tensor = helper.make_tensor(
        name="target_shape",
        data_type=TensorProto.INT64,
        dims=[2],
        vals=[1, 1]
    )
    
    constant_shape_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["shape_tensor_out"],
        value=shape_tensor
    )

    # Node 2: Reshape del tensor reduced a [1, 1] usando shape_tensor_out
    reshape_node = helper.make_node(
        "Reshape",
        inputs=["reduced", "shape_tensor_out"],
        outputs=["reshaped"]
    )

    # Node 3: Aplicar Sigmoid para obtener el score en rango 0.0 a 1.0
    sigmoid_node = helper.make_node(
        "Sigmoid",
        inputs=["reshaped"],
        outputs=["confidence_score"]
    )

    # Crear el grafo (GraphProto)
    graph_def = helper.make_graph(
        nodes=[reduce_node, constant_shape_node, reshape_node, sigmoid_node],
        name="DummyCNNClassifier",
        inputs=[cam_input],
        outputs=[confidence_score]
    )

    # Crear el modelo (ModelProto) especificando opset_imports (opset 12)
    opset = helper.make_operatorsetid("", 12)
    model_def = helper.make_model(graph_def, producer_name="onnx-dummy-generator", opset_imports=[opset])

    # Verificar el modelo para asegurar su validez
    try:
        onnx.checker.check_model(model_def)
        print("El modelo dummy ONNX es válido.")
    except Exception as e:
        print(f"Error en la validación del modelo: {e}")
        return False

    # Guardar el modelo
    onnx.save(model_def, output_path)
    print(f"Modelo dummy ONNX guardado exitosamente en: {output_path}")
    return True

if __name__ == "__main__":
    create_dummy_onnx_model()
