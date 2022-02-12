/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ts;

import java.util.ArrayList;
import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Scanner;

/**
 *
 * @author ACER
 */
public class Data {

    public int numberOfGroup;
    public int numberOfCandidate;
    public int numberOfSkill;
    public ArrayList<Integer> groupNeed;
    public double[][] candidateSkill;
    public int[][] neededSkill;
    public double[] maxPointDeep;
    public double[] minPointWise;
    public double[] maxPointWise;
    public double[] minPointDeep;
    public double[] dw;
    public double[] ww;


    public Data(int numberOfGroup, int numberOfCandidate, int numberOfSkill, ArrayList<Integer> groupNeed, double[][] candidateSkill) {
        this.numberOfGroup = numberOfGroup;
        this.numberOfCandidate = numberOfCandidate;
        this.numberOfSkill = numberOfSkill;
        this.groupNeed = groupNeed;
        this.candidateSkill = candidateSkill;
    }

    public Data() {
    }

    public static Data readDataFromFile() throws FileNotFoundException, IOException {
        Data data = new Data();
        data.numberOfCandidate = 500;
        data.numberOfGroup = 3;
        data.numberOfSkill = 37;
        data.candidateSkill = new double[data.numberOfCandidate][data.numberOfSkill];
        double[] dw = {0.2,0.15,0.15};
        data.ww=dw;
        data.dw= dw;

        ArrayList<Integer> g = new ArrayList<>();
        g.add(2);
        g.add(2);
        g.add(1);
        data.groupNeed = g;
        int[][] arr = {{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
        {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1},
        {0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1}};
        data.neededSkill = arr;
        StringBuffer sb = new StringBuffer();
        BufferedReader br = Files.newBufferedReader(Paths.get("C:\\Users\\ACER\\Downloads\\Gs-master\\gas\\asia.txt"));
        String line;
        String[] arrString = null;

        for (int i = 0; i < data.numberOfCandidate; i++) {
            line = br.readLine();
            arrString = line.split(";");
            for (int j = 0; j < data.numberOfSkill; j++) {
                data.candidateSkill[i][j] = Double.parseDouble(arrString[j]);
            }
        }

        data.maxPointDeep = new double[3];
        data.maxPointWise = new double[3];
        data.minPointWise = new double[3];
        data.minPointDeep = new double[3];

        double[][] transpose = transposeMatrix(data.candidateSkill);

        double[][] res = new double[data.numberOfSkill][data.numberOfCandidate];
        for (int s = 0; s < data.numberOfSkill; s++) {
            double[] g4 = Arrays.copyOf(transpose[s], transpose[s].length);
            Arrays.sort(g4);
            res[s] = g4;
        }
        
        for (int gr = 0; gr < data.numberOfGroup; gr++) {
            double a = 0;
            double b = 0;
            double c = 0;
            double d = 0;

            for (int s = 0; s < data.numberOfSkill; s++) {
                for (int k = 0; k < data.groupNeed.get(gr); k++) {
                    a += data.neededSkill[gr][s] * res[s][data.numberOfCandidate - k-1];
                    b += data.neededSkill[gr][s] * res[s][k];
                    c += Double.min(1.0, res[s][data.numberOfCandidate - k - 1]);
                    d += Double.min(1.0, data.neededSkill[gr][k] * res[s][k]);
                }

            }
            data.maxPointDeep[gr] = a;
            data.minPointDeep[gr] = b;
            data.maxPointWise[gr] = c;
            data.minPointWise[gr] = d;

        }
        
        return data;
    }

    public static double[][] transposeMatrix(double[][] matrix) {
        int m = matrix.length;
        int n = matrix[0].length;

        double[][] transposedMatrix = new double[n][m];

        for (int x = 0; x < n; x++) {
            for (int y = 0; y < m; y++) {
                transposedMatrix[x][y] = matrix[y][x];
            }
        }

        return transposedMatrix;
    }

}
