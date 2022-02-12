/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ts;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Random;

/**
 *
 * @author ACER
 */
public class TS {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws FileNotFoundException, IOException {
        Data data = new Data();
        data = Data.readDataFromFile();
        Solution s = new Solution(data);
        ArrayList<ArrayList<Integer>> can = new ArrayList<>();
        ArrayList<Integer> c = new ArrayList<>();
        c.add(6);
        c.add(0);
        can.add(c);
        c = new ArrayList<>();
        c.add(5);
        c.add(456);
        can.add(c);
        c = new ArrayList<>();
        c.add(330);
        can.add(c);
        s.candi = can;
        System.out.println(s.callFitness(data));
         Problem problem = new Problem(data);
        GA genetic = new GA(problem, data);

        ArrayList<Solution> parto =  genetic.Search();
          BufferedWriter writer2 = new BufferedWriter(new FileWriter("output2.txt"));
         writer2.write("Fitness ; Time Finish ; Total Salary ; Total Exper \n");
       
        for (Solution individual:parto){
            problem.cal_objectives(individual);
            writer2.write(individual.callFitness(data)+ ";" + individual.objectives[0] +";" + individual.objectives[1]+ " ;" + individual.objectives[2]+  ";" + individual.objectives[3] +";" + individual.objectives[4]+ " ;" + individual.objectives[5]+"\n");
        }
        FileOutputStream out = new FileOutputStream(new File("output_parto1.txt"));

        out.close();
    
    }

}
