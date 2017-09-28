import java.util.*;
import java.io.*;
import java.net.*;
import java.nio.file.*;

/*******************************************************************
 * A class to split the training data into training and validation set
 * The original dataset in 
 * http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
 * only provides training and test set (no validation)
 */
public class Preprocess {
	public static final String f_currPathStr = Paths.get("").toAbsolutePath().toString();
	public static final String f_datasetFolder = "/aclImdb";
	public static final double f_percentageValid = 0.2;

	public static void createValidFolder() {
		File folderPos =  new File(f_currPathStr + f_datasetFolder + "/valid/pos/");
		folderPos.mkdirs();
		File folderNeg =  new File(f_currPathStr + f_datasetFolder + "/valid/neg/");
		folderNeg.mkdirs();
	}

	public static void copyFileUsingJava7Files(File source, File dest) {
		try {
			Files.copy(source.toPath(), dest.toPath());
		}
		catch (Exception e) {
			System.err.println("GenericHelper.java : copyFileUsingJava7Files() Exception caught! Error while copying files");
			// e.printStackTrace();
		}
	}

	// Percentage valid is the amount of data to be taken from the training set as the validation set
	public static void randomSampling(double percentageValid, String type) {
		File folder = new File(f_currPathStr + f_datasetFolder + "/train/" + type);
		File[] listOfFiles = folder.listFiles();

		ArrayList<String> filenames = new ArrayList<String>();

		for (int i = 0; i < listOfFiles.length; i++) {
			if (listOfFiles[i].isFile()) {
				String filename = listOfFiles[i].getName();
				String[] tokens = filename.split("\\.(?=[^\\.]+$)");
				String filenumber = tokens[0];
				
				if (tokens[1].equalsIgnoreCase("txt")) {
					filenames.add(filenumber);
				}
			}
		}
		Collections.shuffle(filenames);
		int indexLimit = (int)(percentageValid * (double) filenames.size());
		for (int i=0; i<indexLimit; i++) {
			String filename = filenames.get(i);
			File source = new File(f_currPathStr + f_datasetFolder + "/train/" + type + "/" + filename + ".txt");
			File dest = new File(f_currPathStr + f_datasetFolder + "/valid/" + type + "/" + filename + ".txt");
			copyFileUsingJava7Files(source, dest);
			System.out.println(filename + " deleted");
			File toBeDeletedFile = new File (f_currPathStr + f_datasetFolder + "/train/" + type + "/" + filename + ".txt");
			toBeDeletedFile.delete();
		}
	}

	public static void main (String[] args) {
		createValidFolder();
		randomSampling(f_percentageValid, "pos");
		randomSampling(f_percentageValid, "neg");
	}
}
