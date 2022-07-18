# ViSA
Vietnamese sentiment analysis 


## <div align="center">🎓Training🎓</div>
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
## <div align="center">🥇Performances🥇</div>
All experiments were performed on an **RTX 3090** with **24GB VRAM**, and  a CPU **AMD EPYC 7282 16-Core Processor** with **64GB RAM**, both of which are available for rent on **[vast.ai](https://vast.ai/)**. The pretrained-model used for comparison are available on **[HuggingFace](https://huggingface.co/models)**.
<details>
    <summary style="font-size: 1.50em; font-weight: bold;">UIT-ViSD4SA (update 18/07/2022)</summary>
    <div align="center"><b>Table 1</b>: The overall experimental results</div>
    <table align="center">
        <thead>
            <tr class="hide_border">
                <th align="left" rowspan="2">Model</th>
                <th align="center" rowspan="2">Accuracy</th>
                <th align="center" colspan="3">micro-Average</th>
                <th align="center" colspan="3">micro-Average</th>
                <th align="center" rowspan="2">Reference</th>
            </tr>
            <tr class="hide_border">
                <th align="center">Precision</th>
                <th align="center">Recall</th>
                <th align="center">F1-score</th>
                <th align="center">Precision</th>
                <th align="center">Recall</th>
                <th align="center">F1-score</th>
            </tr>
        </thead>
        <tbody>
            <tr class="hide_border"><td align="center" colspan="9"><b>Aspect</b></td></tr>
            <tr class="hide_border hide_bottom_border">
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
            <tr class="hide_border hide_bottom_border">
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
            <tr class="hide_border">
                <td align="left">HierRoBERTa_SL</td>
                <td align="center">0.8061</td>
                <td align="center">0.6481</td>
                <td align="center">0.6726</td>
                <td align="center"><b style="color: red">0.6601</b></td>
                <td align="center">0.6169</td>
                <td align="center">0.6509</td>
                <td align="center"><b style="color: red">0.6331</b></td>
                <td align="center">
                    <a href="./statics/logs/hier_roberta_sl.log"><b>Log</b></a>
                </td>
            </tr>
            <tr class="hide_border"><td align="center" colspan="9"><b>Polarity</b></td></tr>
            <tr class="hide_border hide_bottom_border">
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
            <tr class="hide_border hide_bottom_border">
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
            <tr class="hide_border">
                <td align="left">HierRoBERTa_SL</td>
                <td align="center">0.8110</td>
                <td align="center">0.6464</td>
                <td align="center">0.6659</td>
                <td align="center"><b style="color: red">0.6560</b></td>
                <td align="center">0.5601</td>
                <td align="center">0.5747</td>
                <td align="center"><b style="color: red">0.5673</b></td>
                <td align="center">
                    <a href="./statics/logs/hier_roberta_sl.log"><b>Log</b></a>
                </td>
            </tr>
            <tr class="hide_border"><td align="center" colspan="9"><b>Aspect-polarity</b></td></tr>
            <tr class="hide_border hide_bottom_border">
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
            <tr class="hide_border hide_bottom_border">
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
            <tr class="hide_border">
                <td align="left">HierRoBERTa_SL</td>
                <td align="center">0.7709</td>
                <td align="center">0.6128</td>
                <td align="center">0.6401</td>
                <td align="center"><b style="color: red">0.6262</b></td>
                <td align="center">0.5089</td>
                <td align="center">0.5389</td>
                <td align="center"><b style="color: red">0.5166</b></td>
                <td align="center">
                    <a href="./statics/logs/hier_roberta_sl.log"><b>Log</b></a>
                </td>
            </tr>
        </tbody>
    </table>
    <div align="center"><b>Table 2</b>: Result per class for aspect label.</div>
    <table align="center">
        <thead>
             <tr class="hide_border">
               <th align="left" rowspan="2">Aspect</th>
                <th align="center" colspan="3">General Scores</th>
                <th align="center" colspan="3">Polarity F1-scores</th>
            </tr>
            <tr class="hide_border">
                <th align="center">Precision</th>
                <th align="center">Recall</th>
                <th align="center">F1-score</th>
                <th align="center">Negative</th>
                <th align="center">Neutral</th>
                <th align="center">Positive</th>
            </tr>
        </thead>
        <tbody>
            <tr class="hide_border hide_bottom_border">
                <td align="left">BATTERY</td>
                <td align="center">0.7507</td>
                <td align="center">0.7621</td>
                <td align="center">0.7563</td>
                <td align="center">0.5900</td>
                <td align="center">0.4387</td>
                <td align="center">0.7907</td>
            </tr>
            <tr class="hide_border hide_bottom_border">
                <td align="left">CAMERA</td>
                <td align="center">0.7563</td>
                <td align="center">0.7796</td>
                <td align="center"><b style="color: red">0.7678</b></td>
                <td align="center">0.5934</td>
                <td align="center">0.5578</td>
                <td align="center"><b style="color: red">0.8179</b></td>
            </tr>
            <tr class="hide_border hide_bottom_border">
                <td align="left">DESIGN</td>
                <td align="center">0.6891</td>
                <td align="center">0.7244</td>
                <td align="center">0.7063</td>
                <td align="center">0.4821</td>
                <td align="center">0.1481</td>
                <td align="center">0.7677</td>
            </tr>
            <tr class="hide_border hide_bottom_border">
                <td align="left">FEATURES</td>
                <td align="center">0.5744</td>
                <td align="center">0.5723</td>
                <td align="center">0.5733</td>
                <td align="center">0.5081</td>
                <td align="center">0.4615</td>
                <td align="center">0.6591</td>
            </tr>
            <tr class="hide_border hide_bottom_border">
                <td align="left">GENERAL</td>
                <td align="center">0.6636</td>
                <td align="center">0.6607</td>
                <td align="center">0.6621</td>
                <td align="center">0.5498</td>
                <td align="center">0.4627</td>
                <td align="center">0.6677</td>
            </tr>
            <tr class="hide_border hide_bottom_border">
                <td align="left">PERFORMANCE</td>
                <td align="center">0.6077</td>
                <td align="center">0.6557</td>
                <td align="center">0.6308</td>
                <td align="center">0.4758</td>
                <td align="center">0.3087</td>
                <td align="center">0.6973</td>
            </tr>
            <tr class="hide_border hide_bottom_border">
                <td align="left">PRICE</td>
                <td align="center">0.4647</td>
                <td align="center">0.4826</td>
                <td align="center">0.4735</td>
                <td align="center">0.3520</td>
                <td align="center">0.2576</td>
                <td align="center">0.5243</td>
            </tr>
            <tr class="hide_border hide_bottom_border">
                <td align="left">SCREEN</td>
                <td align="center">0.6069</td>
                <td align="center">0.6993</td>
                <td align="center">0.6498</td>
                <td align="center">0.4872</td>
                <td align="center">0.3158</td>
                <td align="center">0.7529</td>
            </tr>
            <tr class="hide_border hide_bottom_border">
                <td align="left">SER&ACC </td>
                <td align="center">0.5820</td>
                <td align="center">0.6431</td>
                <td align="center">0.6111</td>
                <td align="center">0.3302</td>
                <td align="center">0.3077</td>
                <td align="center">0.6743</td>
            </tr>
            <tr class="hide_border">
                <td align="left">STORAGE</td>
                <td align="center">0.4737</td>
                <td align="center">0.5294</td>
                <td align="center">0.5000</td>
                <td align="center">0.2857</td>
                <td align="center">0.5455</td>
                <td align="center">0.6875</td>
            </tr>
        </tbody>
    </table>
    <div align="center"><b>Table 3</b>:  Result per class for only sentiment polarity label</div>
    <table align="center">
        <thead>
            <tr class="hide_border">
                <th align="left">Sentiment</th>
                <th align="center">Precision</th>
                <th align="center">Recall</th>
                <th align="center">F1-score</th>
            </tr>
        </thead>
        <tbody>
            <tr class="hide_border hide_bottom_border">
                <td align="left">NEGATIVE</td>
                <td align="center">0.5409</td>
                <td align="center">0.5601</td>
                <td align="center">0.5503</td>
            </tr>           
            <tr class="hide_border hide_bottom_border">
                <td align="left">NEUTRAL</td>
                <td align="center">0.4151</td>
                <td align="center">0.4181</td>
                <td align="center">0.4166</td>
            </tr>
            <tr class="hide_border hide_bottom_border">
                <td align="left">POSITIVE</td>
                <td align="center">0.7243</td>
                <td align="center">0.7459</td>
                <td align="center"><b style="color: red">0.7350</b></td>
            </tr>
        </tbody>
    </table>
</details>

#tr.hide_border > td {border-left-style: hidden;border-right-style: hidden}
#tr.hide_border > th {border-left-style: hidden;border-right-style: hidden}
tr.hide_bottom_border > td {border-bottom-style: hidden}