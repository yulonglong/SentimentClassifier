public class Preprocess {
	public static final String f_currPathStr = Paths.get("").toAbsolutePath().toString();
	public static final String f_datasetFolder = "/aclImdb";

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

	public static void randomSampling(int numSamples, String type) {
		File folderPos = new File(f_currPathStr + f_datasetFolder + "/train/" + type);
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
		// Do some randomization here
		for (String filename : filenames) {
			File source = new File(f_currPathStr + f_datasetFolder + "/train/" + type + "/" + filename + ".txt");
			File dest = new File(f_currPathStr + f_datasetFolder + "/valid/" + type + "/" + filename + ".txt");
			copyFileUsingJava7Files(source, dec);
			System.out.println(filename + " deleted");
			File toBeDeletedFile = new File (f_currPathStr + f_datasetFolder + "/train/" + type + "/" + filenumber + ".txt");
			toBeDeletedFile.delete();
		}
	}
}
