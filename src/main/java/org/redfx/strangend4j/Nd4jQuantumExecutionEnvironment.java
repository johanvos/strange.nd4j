package org.redfx.strangend4j;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Consumer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.redfx.strange.BlockGate;
import org.redfx.strange.Complex;
import org.redfx.strange.Gate;
import org.redfx.strange.Program;
import org.redfx.strange.QuantumExecutionEnvironment;
import org.redfx.strange.Qubit;
import org.redfx.strange.Result;
import org.redfx.strange.Step;
import org.redfx.strange.gate.Identity;
import org.redfx.strange.gate.Oracle;
import org.redfx.strange.gate.PermutationGate;
import org.redfx.strange.gate.ProbabilitiesGate;
import org.redfx.strange.gate.ThreeQubitGate;
import org.redfx.strange.gate.TwoQubitGate;
import org.redfx.strange.local.Computations;

public class Nd4jQuantumExecutionEnvironment implements QuantumExecutionEnvironment {

    @Override
    public Result runProgram(Program p) {
        int nQubits = p.getNumberQubits();
        Qubit[] qubit = new Qubit[nQubits];
        for (int i = 0; i < nQubits; i++) {
            qubit[i] = new Qubit();
        }
        int dim = 1 << nQubits;
        double[] initalpha = p.getInitialAlphas();
        Complex[] probs = new Complex[dim];
        for (int i = 0; i < dim; i++) {
            probs[i] = Complex.ONE;
            for (int j = 0; j < nQubits; j++) {
                int pw = nQubits - j -1 ;
                int pt = 1 << pw;
                int div = i/pt; 
                int md = div % 2;
                if (md == 0) { 
                    probs[i] = probs[i].mul(initalpha[j]);
                } else {
                    probs[i] = probs[i].mul(Math.sqrt(1-initalpha[j]*initalpha[j]));
                }
            }
        }
        List<Step> steps = p.getSteps();
        List<Step> simpleSteps = p.getDecomposedSteps();
        if (simpleSteps == null) { 
            simpleSteps = new ArrayList<>();
            for (Step step : steps) {
                simpleSteps.addAll(Computations.decomposeStep(step, nQubits));
            }
            p.setDecomposedSteps(simpleSteps);
        }
        Result result = new Result(nQubits, steps.size());
        int cnt = 0;
        if (simpleSteps.isEmpty()) {
            result.setIntermediateProbability(0, probs); 
        }
        for (Step step : simpleSteps) {
            if (!step.getGates().isEmpty()) {
                cnt++;
                probs = applyStep(step, probs, qubit);
                int idx = step.getComplexStep();
                if (idx > -1) {
                    result.setIntermediateProbability(idx, probs);
                }
            }
        }
        double[] qp = calculateQubitStatesFromVector(probs);
        for (int i = 0; i < nQubits; i++) {
            qubit[i].setProbability(qp[i]);
        }
        result.measureSystem();
        p.setResult(result);
        return result;
    }

    static Complex[] permutateVector(Complex[] vector, int a, int b) {
        int amask = 1 << a;
        int bmask = 1 << b;
        int dim = vector.length;
        Complex[] answer = new Complex[dim];
        for (int i = 0; i < dim; i++) {
            int j = i;
            int x = (amask & i) /amask;
            int y = (bmask & i) /bmask;
            if (x != y) {
               j ^= amask;
               j ^= bmask;
            }
            answer[i] = vector[j];
        }
        return answer;
    }

    private Complex[]  applyStep (Step step, Complex[] vector, Qubit[] qubits) {
        long s0 = System.currentTimeMillis();
        List<Gate> gates = step.getGates();
        if (!gates.isEmpty() && gates.get(0) instanceof ProbabilitiesGate ) {
            return vector;
        }
        if (gates.size() == 1 && gates.get(0) instanceof PermutationGate) {
            PermutationGate pg = (PermutationGate)gates.get(0);
            return permutateVector (vector, pg.getIndex1(), pg.getIndex2());
        }

        Complex[] result = new Complex[vector.length];
        result = calculateNewState(gates, vector, qubits.length);
        long s1 = System.currentTimeMillis();
        return result;
    }
    
    public Complex[][] mmul(Complex[][] a, Complex[][] b) {
        System.err.println("ND4J mul");
            long l0 = System.currentTimeMillis();
        int arow = a.length;
        int acol = a[0].length;
        int brow = b.length;
        int bcol = b[0].length;
        int am = 0;
        if (acol != brow) {
            throw new RuntimeException("#cols a " + acol + " != #rows b " + brow);
        }
        Complex[][] answer = new Complex[arow][bcol];
        double[][] ar = new double[arow][acol];
        double[][] ai = new double[arow][acol];
        double[][] br = new double[brow][bcol];
        double[][] bi = new double[brow][bcol];
        for (int i = 0; i < arow; i++) {
            for (int j = 0; j < bcol; j++) {
                Complex el = new Complex(0., 0.);
                double newr = 0;
                double newi = 0;
                boolean zero = true;
                for (int k = 0; k < acol; k++) {
                    if (j == 0) {
                        ar[i][k] = a[i][k].r;
                        ai[i][k] = a[i][k].i;
                    }
                    if (i == 0) {
                        br[k][j] = b[k][j].r;
                        bi[k][j] = b[k][j].i;
                    }
                }
                if (zero) {
                    answer[i][j] = Complex.ZERO;
                } else {
                    answer[i][j] = Complex.ZERO;
                }


            }
        }
        long l1 = System.currentTimeMillis();
        INDArray n_ar = Nd4j.create(ar);
        INDArray n_ai = Nd4j.create(ai);
        INDArray n_br = Nd4j.create(br);
        INDArray n_bi = Nd4j.create(bi);
        INDArray n_r = n_ar.mmul(n_br).sub(n_ai.mmul(n_bi));
        INDArray n_i = n_ai.mmul(n_br).add(n_ar.mmul(n_bi));

        for (int i = 0; i < acol; i++) {
            for (int j = 0; j < brow; j++) {
                answer[i][j] = new Complex(n_r.getDouble(i, j), n_i.getDouble(i, j));

            }
        }
        return answer;
    }
    
    private double[] calculateQubitStatesFromVector(Complex[] vectorresult) {
        int nq = (int) Math.round(Math.log(vectorresult.length) / Math.log(2));
        double[] answer = new double[nq];
        int ressize = 1 << nq;
        for (int i = 0; i < nq; i++) {
            int pw = i;//nq - i - 1;
            int div = 1 << pw;
            for (int j = 0; j < ressize; j++) {
                int p1 = j / div;
                if (p1 % 2 == 1) {
                    answer[i] = answer[i] + vectorresult[j].abssqr();
                }
            }
        }
        return answer;
    }

    @Override
    public void runProgram(Program p, Consumer<Result> result) {
        Thread t = new Thread(() -> result.accept(runProgram(p)));
        t.start();
    }
         
    private static Complex[][] fastmmul(Complex[][] a, Complex[][] b) {
        long l0 = System.currentTimeMillis();
        int arow = a.length;
        int acol = a[0].length;
        int brow = b.length;
        int bcol = b[0].length;
        int am = 0;
        if (acol != brow) {
            throw new RuntimeException("#cols a " + acol + " != #rows b " + brow);
        }
        Complex[][] answer = new Complex[arow][bcol];
        double[][] ar = new double[arow][acol];
        double[][] ai = new double[arow][acol];
        double[][] br = new double[brow][bcol];
        double[][] bi = new double[brow][bcol];
        for (int i = 0; i < arow; i++) {
            for (int j = 0; j < bcol; j++) {
                Complex el = new Complex(0., 0.);
                double newr = 0;
                double newi = 0;
                boolean zero = true;
                for (int k = 0; k < acol; k++) {
                    if (j == 0) {
                        ar[i][k] = a[i][k].r;
                        ai[i][k] = a[i][k].i;
                    }
                    if (i == 0) {
                        br[k][j] = b[k][j].r;
                        bi[k][j] = b[k][j].i;
                    }
                }
                if (zero) {
                    answer[i][j] = Complex.ZERO;
                } else {
                    answer[i][j] = Complex.ZERO;
                }

            }
        }
        //     if (1 < 2) System.exit(0);
        long l1 = System.currentTimeMillis();
        INDArray n_ar = Nd4j.create(ar);
        INDArray n_ai = Nd4j.create(ai);
        INDArray n_br = Nd4j.create(br);
        INDArray n_bi = Nd4j.create(bi);
        INDArray n_r = n_ar.mmul(n_br).sub(n_ai.mmul(n_bi));
        INDArray n_i = n_ai.mmul(n_br).add(n_ar.mmul(n_bi));

        for (int i = 0; i < acol; i++) {
            for (int j = 0; j < brow; j++) {
                answer[i][j] = new Complex(n_r.getDouble(i, j), n_i.getDouble(i, j));
            }
        }
        return answer;

    }

    
    private static List<Gate> getAllGates(List<Gate> gates, int nQubits) {
        List<Gate> answer = new ArrayList<>();
        int idx = nQubits -1;
          while (idx >= 0) {
            final int cnt = idx;
            Gate myGate = gates.stream()
                    .filter(
                        gate -> gate.getHighestAffectedQubitIndex() == cnt )
                    .findFirst()
                    .orElse(new Identity(idx));
                           answer.add(myGate);    
           if (myGate instanceof BlockGate) {
                BlockGate sqg = (BlockGate)myGate;
                idx = idx - sqg.getSize()+1;
            }           
            if (myGate instanceof TwoQubitGate) {
                idx--;
            }
            if (myGate instanceof ThreeQubitGate) {
                idx = idx-2;
            }
            if (myGate instanceof PermutationGate) {
                throw new RuntimeException("No perm allowed ");
            }
            if (myGate instanceof Oracle) {
                idx = 0;
            }
            idx--;
        }
        return answer;
    }

    public Complex[] calculateNewState(List<Gate> gates, Complex[] vector, int length) {
        return getNextProbability(getAllGates(gates, length), vector);
    }
    
    private Complex[] getNextProbability(List<Gate> gates, Complex[] v) {
         Gate gate = gates.get(0);

        Complex[][] matrix = gate.getMatrix(this);
        int size = v.length;

        if (gates.size() > 1) {
            List<Gate> nextGates = gates.subList(1, gates.size());
            int gatedim = matrix.length;
            int partdim = size/gatedim;
            Complex[] answer = new Complex[size];
            Complex[][] vsub = new Complex[gatedim][partdim];
            for (int i = 0; i < gatedim; i++) {
                Complex[] vorig = new Complex[partdim];
                for (int j = 0; j < partdim; j++) {
                    vorig[j] = v[j + i *partdim];
                }
                vsub[i] = getNextProbability(nextGates, vorig);
            }
            for (int i = 0; i < gatedim; i++) {
                for (int j = 0; j < partdim; j++) {
                    answer[j + i * partdim] = Complex.ZERO;
                    for (int k = 0; k < gatedim;k++) {
                        answer[j + i * partdim] = answer[j + i * partdim].add(matrix[i][k].mul(vsub[k][j]));
                    }
                }
            }
            return answer;
        } else {
            if (matrix.length != size) {
                System.err.println("problem with matrix for gate "+gate);
                throw new IllegalArgumentException ("wrong matrix size "+matrix.length+" vs vector size "+v.length);
            }
            Complex[] answer = new Complex[size];
            for (int i = 0; i < size; i++) {
                answer[i] = Complex.ZERO;
                for (int j = 0; j < size; j++) {
                    answer[i] = answer[i].add(matrix[i][j].mul(v[j]));
                }
            }
            return answer;
        }
    }
           
}
