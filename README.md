# ViSA
Vietnamese sentiment analysis 


## 🎓 Training
The commands below **train**/**fine-tune** model for **Sentiment analysis**.
```bash
python main.py train --task UIT-ViSD4SA \
                     --model_arch hier_roberta_sl \
                     --run_test \
                     --data_dir datasets/UIT-ViSD4SA \
                     --model_name_or_path vinai/phobert-base \
                     --output_dir outputs \
                     --max_seq_length 256 \
                     --train_batch_size 32 \
                     --eval_batch_size 32 \
                     --learning_rate 1e-4 \
                     --classifier_learning_rate 3e-3 \
                     --epochs 100 \
                     --early_stop 50 \
                     --overwrite_data
```
## 🥇 Performances
All experiments were performed on an **RTX 3090** with **24GB VRAM**, and  a CPU **AMD EPYC 7282 16-Core Processor** with **64GB RAM**, both of which are available for rent on **[vast.ai](https://vast.ai/)**. The pretrained-model used for comparison are available on **[HuggingFace](https://huggingface.co/models)**.
<details>
    <summary style="font-size: 1.50em; font-weight: bold;">UIT-ViSD4SA (update 18/07/2022)</summary>
    <div align="center"><b>Table 1</b>: The overall experimental results</div>
    <table align="center">
        <thead>
            <tr>
                <th align="left" rowspan="2">Model</th>
                <th align="center" rowspan="2">Accuracy</th>
                <th align="center" colspan="3">micro-Average</th>
                <th align="center" colspan="3">macro-Average</th>
                <th align="center" rowspan="2">Reference</th>
            </tr>
            <tr>
                <th align="center">Precision</th>
                <th align="center">Recall</th>
                <th align="center">F1-score</th>
                <th align="center">Precision</th>
                <th align="center">Recall</th>
                <th align="center">F1-score</th>
            </tr>
        </thead>
        <tbody>
            <tr><td align="center" colspan="9"><b>Aspect</b></td></tr>
            <tr>
                <td align="left">BiLSTM_CRF_Base</td>
                <td align="center">.....</td>
                <td align="center">0.6563</td>
                <td align="center">0.6515</td>
                <td align="center">0.6539</td>
                <td align="center">0.6288</td>
                <td align="center">0.6162</td>
                <td align="center">0.6217</td>
                <td align="center">
                    <a href="https://aclanthology.org/2021.paclic-1.34.pdf"><b>Paper</b></a>
                </td>
            </tr>
            <tr>
                <td align="left">BiLSTM_CRF_Large</td>
                <td align="center">.....</td>
                <td align="center">0.6496</td>
                <td align="center">0.6685</td>
                <td align="center">0.6589</td>
                <td align="center">0.6200</td>
                <td align="center">0.6356</td>
                <td align="center">0.6276</td>
                <td align="center">
                    <a href="https://aclanthology.org/2021.paclic-1.34.pdf"><b>Paper</b></a>
                </td>
            </tr>
            <tr>
                <td align="left">HierRoBERTa_SL</td>
                <td align="center">0.8061</td>
                <td align="center">0.6481</td>
                <td align="center">0.6726</td>
                <td align="center">0.6601</td>
                <td align="center">0.6169</td>
                <td align="center">0.6509</td>
                <td align="center">0.6331</td>
                <td align="center">
                    <a href="./statics/logs/hier_roberta_sl.log"><b>Log</b></a>
                </td>
            </tr>
            <tr>
                <td align="left">HierRoBERTa_ML</td>
                <td align="center">0.8045</td>
                <td align="center">0.6528</td>
                <td align="center">0.6750</td>
                <td align="center"><b style="color: red">0.6637</b></td>
                <td align="center">0.6324</td>
                <td align="center">0.6474</td>
                <td align="center"><b style="color: red">0.6391</b></td>
                <td align="center">
                    <a href="./statics/logs/hier_roberta_ml.log"><b>Log</b></a>
                </td>
            </tr>
            <tr><td align="center" colspan="9"><b>Polarity</b></td></tr>
            <tr>
                <td align="left">BiLSTM_CRF_Base</td>
                <td align="center">.....</td>
                <td align="center">0.5488 </td>
                <td align="center">0.5591</td>
                <td align="center">0.5539</td>
                <td align="center">0.4687</td>
                <td align="center">0.4639</td>
                <td align="center">0.4657</td>
                <td align="center"><a href="https://aclanthology.org/2021.paclic-1.34.pdf"><b>Paper</b></a></td>
            </tr>
            <tr>
                <td align="left">BiLSTM_CRF_Large</td>
                <td align="center">.....</td>
                <td align="center">0.5689 </td>
                <td align="center">0.5978</td>
                <td align="center">0.5830</td>
                <td align="center">0.4900</td>
                <td align="center">0.5060</td>
                <td align="center">0.4977</td>
                <td align="center"><a href="https://aclanthology.org/2021.paclic-1.34.pdf"><b>Paper</b></a></td>
            </tr>
            <tr>
                <td align="left">HierRoBERTa_SL</td>
                <td align="center">0.8110</td>
                <td align="center">0.6464</td>
                <td align="center">0.6659</td>
                <td align="center">0.6560</td>
                <td align="center">0.5601</td>
                <td align="center">0.5747</td>
                <td align="center">0.5673</td>
                <td align="center">
                    <a href="./statics/logs/hier_roberta_sl.log"><b>Log</b></a>
                </td>
            </tr>
            <tr>
                <td align="left">HierRoBERTa_ML</td>
                <td align="center">0.8085</td>
                <td align="center">0.6526</td>
                <td align="center">0.6655</td>
                <td align="center"><b style="color: red">0.6590</b></td>
                <td align="center">0.5794</td>
                <td align="center">0.5734</td>
                <td align="center"><b style="color: red">0.5757</b></td>
                <td align="center">
                    <a href="./statics/logs/hier_roberta_ml.log"><b>Log</b></a>
                </td>
            </tr>
            <tr><td align="center" colspan="9"><b>Aspect-polarity</b></td></tr>
            <tr>
                <td align="left">BiLSTM_CRF_Base</td>
                <td align="center">.....</td>
                <td align="center">0.6071</td>
                <td align="center">0.6162</td>
                <td align="center">0.6116</td>
                <td align="center">0.4618</td>
                <td align="center">0.4342</td>
                <td align="center">0.4437</td>
                <td align="center"><a href="https://aclanthology.org/2021.paclic-1.34.pdf"><b>Paper</b></a></td>
            </tr>
            <tr>
                <td align="left">BiLSTM_CRF_Large</td>
                <td align="center">.....</td>
                <td align="center">0.6178</td>
                <td align="center">0.6299</td>
                <td align="center">0.6238</td>
                <td align="center">0.4684</td>
                <td align="center">0.4546</td>
                <td align="center">0.4570</td>
                <td align="center"><a href="https://aclanthology.org/2021.paclic-1.34.pdf"><b>Paper</b></a></td>
            </tr>
            <tr>
                <td align="left">HierRoBERTa_SL</td>
                <td align="center">0.7709</td>
                <td align="center">0.6128</td>
                <td align="center">0.6401</td>
                <td align="center">0.6262</td>
                <td align="center">0.5089</td>
                <td align="center">0.5389</td>
                <td align="center">0.5166</td>
                <td align="center">
                    <a href="./statics/logs/hier_roberta_sl.log"><b>Log</b></a>
                </td>
            </tr>
            <tr>
                <td align="left">HierRoBERTa_ML</td>
                <td align="center">0.7706</td>
                <td align="center">0.6213</td>
                <td align="center">0.6416</td>
                <td align="center"><b style="color: red">0.6313</b></td>
                <td align="center">0.5391</td>
                <td align="center">0.5195</td>
                <td align="center"><b style="color: red">0.5206</b></td>
                <td align="center">
                    <a href="./statics/logs/hier_roberta_ml.log"><b>Log</b></a>
                </td>
            </tr>
        </tbody>
    </table>
    <div align="center"><b>Table 2</b>: Result per class for aspect label of HierRoBERTa_ML</div>
    <table align="center">
        <thead>
             <tr>
               <th align="left" rowspan="2">Aspect</th>
                <th align="center" colspan="3">General Scores</th>
                <th align="center" colspan="3">Polarity F1-scores</th>
            </tr>
            <tr>
                <th align="center">Precision</th>
                <th align="center">Recall</th>
                <th align="center">F1-score</th>
                <th align="center">Negative</th>
                <th align="center">Neutral</th>
                <th align="center">Positive</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td align="left">BATTERY</td>
                <td align="center">0.7511</td>
                <td align="center">0.7612</td>
                <td align="center">0.7561</td>
                <td align="center">0.5944</td>
                <td align="center">0.5231</td>
                <td align="center"><b style="color: red">0.8121</b></td>
            </tr>
            <tr>
                <td align="left">CAMERA</td>
                <td align="center">0.7588</td>
                <td align="center">0.7650</td>
                <td align="center"><b style="color: red">0.7619</b></td>
                <td align="center">0.5836</td>
                <td align="center">0.5823</td>
                <td align="center">0.8062</td>
            </tr>
            <tr>
                <td align="left">DESIGN</td>
                <td align="center">0.7059</td>
                <td align="center">0.7024</td>
                <td align="center">0.7042</td>
                <td align="center">0.4188</td>
                <td align="center">0.2857</td>
                <td align="center">0.7600</td>
            </tr>
            <tr>
                <td align="left">FEATURES</td>
                <td align="center">0.5600</td>
                <td align="center">0.5784</td>
                <td align="center">0.5690</td>
                <td align="center">0.4894</td>
                <td align="center">0.4545</td>
                <td align="center">0.6667</td>
            </tr>
            <tr>
                <td align="left">GENERAL</td>
                <td align="center">0.6537</td>
                <td align="center">0.6743</td>
                <td align="center">0.6638</td>
                <td align="center">0.5478</td>
                <td align="center">0.4685</td>
                <td align="center">0.6705</td>
            </tr>
            <tr>
                <td align="left">PERFORMANCE</td>
                <td align="center">0.6381</td>
                <td align="center">0.6535</td>
                <td align="center">0.6457</td>
                <td align="center">0.5061</td>
                <td align="center">0.2714</td>
                <td align="center">0.7165</td>
            </tr>
            <tr>
                <td align="left">PRICE</td>
                <td align="center">0.4640</td>
                <td align="center">0.4981</td>
                <td align="center">0.4804</td>
                <td align="center">0.3937</td>
                <td align="center">0.2963</td>
                <td align="center">0.4907</td>
            </tr>
            <tr>
                <td align="left">SCREEN</td>
                <td align="center">0.6735</td>
                <td align="center">0.7174</td>
                <td align="center">0.6947</td>
                <td align="center">0.5067</td>
                <td align="center">0.3529</td>
                <td align="center">0.7748</td>
            </tr>
            <tr>
                <td align="left">SER&ACC </td>
                <td align="center">0.5672</td>
                <td align="center">0.6527</td>
                <td align="center">0.6069</td>
                <td align="center">0.2939</td>
                <td align="center">0.2857</td>
                <td align="center">0.6727</td>
            </tr>
            <tr>
                <td align="left">STORAGE</td>
                <td align="center">0.5517</td>
                <td align="center">0.4706</td>
                <td align="center">0.5079</td>
                <td align="center">0.3478</td>
                <td align="center">0.4444</td>
                <td align="center">0.6000</td>
            </tr>
        </tbody>
    </table>
    <div align="center"><b>Table 3</b>:  Result per class for only sentiment polarity label of HierRoBERTa_ML</div>
    <table align="center">
        <thead>
            <tr>
                <th align="left">Sentiment</th>
                <th align="center">Precision</th>
                <th align="center">Recall</th>
                <th align="center">F1-score</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td align="left">NEGATIVE</td>
                <td align="center">0.5400</td>
                <td align="center">0.5579</td>
                <td align="center">0.5488</td>
            </tr>           
            <tr>
                <td align="left">NEUTRAL</td>
                <td align="center">0.4704</td>
                <td align="center">0.4157</td>
                <td align="center">0.4414</td>
            </tr>
            <tr>
                <td align="left">POSITIVE</td>
                <td align="center">0.7278</td>
                <td align="center">0.7466</td>
                <td align="center"><b style="color: red">0.7371</b></td>
            </tr>
        </tbody>
    </table>
</details>
<details>
    <summary style="font-size: 1.50em; font-weight: bold;">ABSA (update 18/07/2022)</summary>
</details>
<details>
    <summary style="font-size: 1.50em; font-weight: bold;">YASO (update 18/07/2022)</summary>
</details>

## 📋 Todo
#### Models
- [x] ~~Implement **Hierarchical RoBERTa model** (***single layer***).~~
- [x] ~~Implement **Hierarchical RoBERTa model** (***multiple layers***).~~
- [ ] Implement **Hierarchical BERT model**.
- [x] ~~Implement **Controlable Task-dependency loss**.~~
#### Dataset processors
- [x] ~~Read the **UIT-ViSD4SA** dataset and convert it to ABSA features.~~
- [ ] Read the **ABSA-**{***laptop***, ***rest***, ***twitter***} dataset and convert it to ABSA features.
- [ ] Read the **YASO** dataset and convert it to ABSA features.
#### Pipelines
- [X] ~~Complete **Train** pipeline.~~
- [X] ~~Complete **Test** pipeline.~~
- [X] Complete **Predict** pipeline.
- [X] ~~Code metrics for evaluate ABSA task.~~
#### Documents
- [ ] Introduce ViSA and its features and implemented model (**Introduction**].
- [ ] Configure the environment and install any necessary libraries (**Environments**).
- [ ] How to execute ViSA to train/fine-tune and validate the model (**Training**).
- [x] ~~Describe the experimental setup and performance of the models (**Performances**).~~

