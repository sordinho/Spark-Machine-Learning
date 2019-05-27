package it.sordinho.spark.sparkmllib;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.feature.LabeledPoint;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

/**
 * Created by Davide Sordi
 * Using IntelliJ IDEA
 * Date: 27/05/19
 * Time: 18.20
 * <p>
 * Class: SparkDriver
 * Project: SparkMllibLogisticRegression
 */
public class SparkDriver {

	public static void main(String[] args) {

		String trainingFile;
		String testFile;
		String outputPath;

		trainingFile = args[0];
		testFile = args[1];
		outputPath = args[2];

		// First we need a spark session
		SparkSession sparkSession = SparkSession.builder().appName("MLLib - logistic regression").getOrCreate();

		// Create a Spark Context
		JavaSparkContext javaSparkContext = new JavaSparkContext(sparkSession.sparkContext());


		/*
		 * ##################################
		 * #        Training step           #
		 * ##################################
		 */

		/*
		 * Read the training data from a text file and store in an RDD of String
		 * Line format: <class-label>,<list of 3 numerical values>
		 */
		JavaRDD<String> trainingData = javaSparkContext.textFile(trainingFile);

		// Map each value of input data to a LabeledPoint
		JavaRDD<LabeledPoint> labeledPointRDD = trainingData.map(line -> {
			// Split records
			String[] fields = line.split(",");

			// Get classLabel as double
			Double classLabel = Double.valueOf(fields[0]);

			// Save the other 3 values in an array
			double[] attributes = new double[3];
			for (int i = 1; i <= 3; i++) {
				attributes[i - 1] = Double.valueOf(fields[i]);
			}

			// Create a dense vector based on a previous array
			Vector attributesValues = Vectors.dense(attributes);

			// Return a new LabeledPoint of the current entry
			return new LabeledPoint(classLabel, attributesValues);
		});

		/*
		 * Preparation of training data.
		 * LabeledPoint is a JavaBean, we convert RDD<JavaBean> to Dataset<Row> with SparkSQL
		 * We cache the dataset due to lazy evaluation
		 */
		Dataset<Row> trainingDataset = sparkSession.createDataFrame(trainingData, LabeledPoint.class).cache();

		/*
		 * Creation of a LogisticRegression object.
		 * LogisticRegression is used to make a classification model
		 */
		LogisticRegression logisticRegression = new LogisticRegression();

		/*
		 * Parameter settings for the logisticRegression
		 * MaxIter : maximum number of the algorithm iterations
		 * RegParam : regularization parameter
		 */
		logisticRegression.setMaxIter(10);
		logisticRegression.setRegParam(0.01);

		/*
		 * Define the pipeline used to create the logistic regression model on training data
		 * In this case the pipeline contains only one stage
		 */
		Pipeline pipeline = new Pipeline().setStages(new PipelineStage[]{logisticRegression});

		// Execute pipeline on training data for creating a model
		PipelineModel model = pipeline.fit(trainingDataset);


		/*
		 * ##################################
		 * #        Prediction step         #
		 * ##################################
		 */

		/*
		 * Now we have a model, we use it to predict class label in this step.
		 * Read the unlabeled data (only attributes are available)
		 */
		JavaRDD<String> unlabeledRawData = javaSparkContext.textFile(testFile);

		// Map each value of input data to a LabeledPoint (this part is the same as in the training step)
		JavaRDD<LabeledPoint> unlabeledData = unlabeledRawData.map(line -> {
			// Split records
			String[] fields = line.split(",");

			// Save the other 3 values in an array
			double[] attributes = new double[3];
			for (int i = 1; i <= 3; i++) {
				attributes[i - 1] = Double.valueOf(fields[i]);
			}

			// Create a dense vector based on a previous array
			Vector attributesValues = Vectors.dense(attributes);

			/*
			 * The label is unknown but we need a class label to create a LabeledPoint.
			 * We set this to -1 (invalid value), this value is not usefull for computation because
			 * it's not used for prediction. (It has to be predicted)
			 */
			double fakeClassLabel = -1.0;

			// Return a new LabeledPoint of the current entry
			return new LabeledPoint(fakeClassLabel, attributesValues);
		});

		// Create a dataset<Row> for  the test data
		Dataset<Row> unlabeledDataset = sparkSession.createDataFrame(unlabeledData, LabeledPoint.class);

		/*
		 * Now we make the prediction using Transformer.transform()
		 *
		 * Returned DataFrame schema:
		 * - features: vector of double (attributes)
		 * - label: double (fake class label)
		 * - rawPrediction:  vector of nullable double (we don't care)
		 * - probability: vector (The i-th cell contains the probability that the current record belongs to the i-th class)
		 * - prediction: double (the predicted value, what we want)
		 */
		Dataset<Row> predictions = model.transform(unlabeledDataset);

		/*
		 * We want to keep only the predicted class label and the attributes
		 * for each record
		 */
		Dataset<Row> cleanPredictions = predictions.select("features","prediction");

		/*
		 * We need now an RDD to save result on a file.
		 * Convert dataset to RDD and save on HDFS file
		 */
		JavaRDD<Row> finalPredictionResult = cleanPredictions.javaRDD();
		finalPredictionResult.saveAsTextFile(outputPath);

		// Close the spark context
		javaSparkContext.close();
	}
}
