import onnx
from onnx import helper
from onnx import TensorProto

def make_dummy_onnx():
    print("[*] Generando modelo ONNX dummy para pruebas...")
    
    # 1. Definir los tensores de entrada y salida con nombres exactos de la pizarra
    input_tensor = helper.make_tensor_value_info('cam_input', TensorProto.FLOAT, [1, 224, 224, 3])
    output_tensor = helper.make_tensor_value_info('confidence_score', TensorProto.FLOAT, [1, 1])

    # 2. Nodo 1: ReduceMean para convertir [1, 224, 224, 3] en [1, 1, 1, 1] (promedio)
    node_mean = helper.make_node(
        'ReduceMean',
        inputs=['cam_input'],
        outputs=['reduced'],
        axes=[1, 2, 3],
        keepdims=1
    )

    # 3. Nodo 2: Reshape para convertir [1, 1, 1, 1] en [1, 1]
    node_reshape = helper.make_node(
        'Reshape',
        inputs=['reduced', 'shape_const'],
        outputs=['confidence_score']
    )

    # 4. Inicializador para la forma de destino [1, 1] en el Reshape
    shape_const = helper.make_tensor('shape_const', TensorProto.INT64, [2], [1, 1])

    # 5. Crear el Grafo
    graph = helper.make_graph(
        [node_mean, node_reshape],
        'dummy_eye_detector',
        [input_tensor],
        [output_tensor],
        [shape_const]
    )

    # 6. Crear el Modelo ONNX e importar opset 12 (especificado en la pizarra como v12+)
    model = helper.make_model(graph, producer_name='dummy_onnx_generator')
    model.opset_import[0].version = 12

    # 7. Validar y Guardar
    onnx.checker.check_model(model)
    onnx.save(model, 'eye_detector.onnx')
    
    print("[+] Modelo ONNX dummy generado con éxito como 'eye_detector.onnx'.")
    print("    - Entrada: 'cam_input' con forma [1, 224, 224, 3]")
    print("    - Salida: 'confidence_score' con forma [1, 1]")

if __name__ == "__main__":
    make_dummy_onnx()
