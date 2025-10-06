{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "3479c93f-9afa-4096-bf20-b17b26037e01",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\CSE\\anaconda3\\envs\\lexenv\\lib\\site-packages\\numpy\\_distributor_init.py:30: UserWarning: loaded more than 1 DLL from .libs:\n",
      "C:\\Users\\CSE\\anaconda3\\envs\\lexenv\\lib\\site-packages\\numpy\\.libs\\libopenblas.FB5AE2TYXYH2IJRDKGDGQ3XBKLKTF43H.gfortran-win_amd64.dll\n",
      "C:\\Users\\CSE\\anaconda3\\envs\\lexenv\\lib\\site-packages\\numpy\\.libs\\libopenblas64__v0.3.21-gcc_10_3_0.dll\n",
      "  warnings.warn(\"loaded more than 1 DLL from .libs:\"\n"
     ]
    }
   ],
   "source": [
    "import re\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "\n",
    "def parse_min_max(s):\n",
    "    if pd.isna(s): return (np.nan, np.nan)\n",
    "    nums = re.findall(r\"[-+]?\\d*\\.\\d+|\\d+\", str(s))\n",
    "    if len(nums) >= 2:\n",
    "        return float(nums[0]), float(nums[1])\n",
    "    return (np.nan, np.nan)\n",
    "\n",
    "def parse_working_value(s):\n",
    "    if pd.isna(s): return np.nan\n",
    "    m = re.search(r\"([-+]?\\d*\\.\\d+|\\d+)\", str(s).replace(\",\", \".\"))\n",
    "    return float(m.group(0)) if m else np.nan\n",
    "\n",
    "def compute_hi_for_row(row, warn_margin=0.1):\n",
    "    val = parse_working_value(row.get(\"WORKING_VALUE_ONBOARD\", np.nan))\n",
    "    low, high = parse_min_max(row.get(\"MIN_MAX_THRESHOLDS\", \"\"))\n",
    "    if np.isnan(val) or np.isnan(low) or np.isnan(high):\n",
    "        return np.nan, \"Unknown\"\n",
    "    if low <= val <= high:\n",
    "        return 1.0, \"Healthy\"\n",
    "    elif (low*(1-warn_margin)) <= val <= (high*(1+warn_margin)):\n",
    "        return 0.5, \"Warning\"\n",
    "    else:\n",
    "        return 0.0, \"Critical\"\n",
    "\n",
    "def evaluate_dataframe(df):\n",
    "    df = df.copy()\n",
    "    df[\"Health_Index\"], df[\"Predicted_Status\"] = zip(*df.apply(compute_hi_for_row, axis=1))\n",
    "    return df\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0fd8b9e8-0c49-40f9-96e8-ce4bde207853",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.19"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
