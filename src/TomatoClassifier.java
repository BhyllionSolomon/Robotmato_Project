import org.deeplearning4j.datasets.datavec.ImageRecordReader;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.MaxPooling2D;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimization.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.learning.config.Adam;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;

import java.io.File;

public class TomatoClassifier {

    public static void main(String[] args) throws Exception {
        // Paths for the dataset (path to the ripe and unripe tomato image directories)
        String dataDir = "path_to_tomato_dataset"; // Add the path to your dataset
        File data = new File(dataDir);
        
        // Define how to load the image data
        ImageRecordReader recordReader = new ImageRecordReader(224, 224, 3, new ParentPathLabelGenerator());
        FileSplit fileSplit = new FileSplit(data);
        recordReader.initialize(fileSplit);
        
        // Create a DataSetIterator for training
        DataSetIterator trainIter = new RecordReaderDataSetIterator(recordReader, 32, 1, 2);
        trainIter.setPreProcessor(new ImagePreProcessingScaler(0, 1));

        // Define the CNN model configuration
        MultiLayerNetwork model = new MultiLayerNetwork(new NeuralNetConfiguration.Builder()
                .seed(12345)
                .updater(new Adam(0.0005))
                .list()
                .layer(0, new ConvolutionLayer.Builder(5, 5)
                        .nIn(3) // Input channels (RGB)
                        .nOut(32) // Output channels
                        .activation(Activation.RELU)
                        .build())
                .layer(1, new MaxPooling2D.Builder()
                        .kernelSize(2, 2)
                        .build())
                .layer(2, new ConvolutionLayer.Builder(5, 5)
                        .nOut(64)
                        .activation(Activation.RELU)
                        .build())
                .layer(3, new MaxPooling2D.Builder()
                        .kernelSize(2, 2)
                        .build())
                .layer(4, new DenseLayer.Builder().nOut(128)
                        .activation(Activation.RELU)
                        .build())
                .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(2) // 2 classes: ripe and unripe
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutionalFlat(224, 224, 3)) // Image size (224x224 RGB)
                .build());

        model.init();
        model.setListeners(new ScoreIterationListener(100));

        // Train the model
        for (int epoch = 0; epoch < 10; epoch++) {
            model.fit(trainIter);
            System.out.println("Epoch " + epoch + " complete.");
        }

        // After training, you can save the model and use it for inference on new images
        model.save(new File("tomato_model.zip"));
    }
}
