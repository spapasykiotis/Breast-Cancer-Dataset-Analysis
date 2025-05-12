import weka.classifiers.trees.J48;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class Classification{

	public static void main(String args[]) throws Exception{
            
	//Load Dataset
		DataSource source = new DataSource("data/breast-cancer-final.arff");
		Instances dataset = source.getDataSet();  
                
        //set class index to first attribute for dataset
                dataset.setClassIndex(0);
                
	//randomize data before split
		dataset.randomize(new java.util.Random(1));
                
	//split the dataset by percentage
		double percentage = 0.5;
		int trainSize = (int) Math.round(dataset.numInstances() * percentage);
		int testSize = dataset.numInstances() - trainSize;
		Instances train = new Instances(dataset, 0, trainSize); //training dataset
		Instances test = new Instances(dataset, trainSize, testSize); //test dataset
                
	//randomize train test dataset before evaluation (Optional)
		test.randomize(new java.util.Random(1));
                
	//set class index to the first attribute for train dataset
		train.setClassIndex(0);
	//create and build the classifier!
		J48 tree = new J48();
		tree.buildClassifier(train);

		Evaluation eval = new Evaluation(train);

	//we build the classifier with the training dataset
        //we initialize evaluation with the training dataset and then
        //evaluate using the test dataset

	//set class index to the first attribute for test dataset
		test.setClassIndex(0);
	//now evaluate model
		eval.evaluateModel(tree, test);
		System.out.println("==================================================================");
		System.out.println(eval.toSummaryString("Classification evaluation results:\n", false));

		//System.out.println("Correct % = "+eval.pctCorrect());
		//System.out.println("Incorrect % = "+eval.pctIncorrect());
		//System.out.println("AUC = "+eval.areaUnderROC(1));
		//System.out.println("kappa = "+eval.kappa());
		//System.out.println("MAE = "+eval.meanAbsoluteError());
		//System.out.println("RMSE = "+eval.rootMeanSquaredError());
		//System.out.println("RAE = "+eval.relativeAbsoluteError());
		//System.out.println("RRSE = "+eval.rootRelativeSquaredError());
		System.out.println("Precision = "+eval.precision(1));
		System.out.println("Recall = "+eval.recall(1));
		System.out.println("fMeasure = "+eval.fMeasure(1));
		System.out.println("Error Rate = "+eval.errorRate());
	//the confusion matrix
		System.out.println(eval.toMatrixString("=== Overall Confusion Matrix ===\n"));
                System.out.println("==================================================================");
	}
}
