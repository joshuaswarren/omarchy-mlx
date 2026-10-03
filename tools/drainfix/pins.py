# DrainFix digest pins: (new_tokens, passes) -> ordered_records_sha256
# d64/d512 passes=1 values reproduced x2 on 2026-10-03 W1 (ctl + straced, ids identical);
# they also match the MesaPack64 receipt A/B table for this wheel. passes=5 values are
# the DecodeBw morning pins (ids bit-identical, more records in the canonical list).
PINS = {
    (64, 1): "cb3e87705c65497cba3614728da06675b94d8dba6fae39a687b7b6f21f35a0fb",
    (512, 1): "619360bf1d624de288d4d8607f0404d410e2e0c34c11bfe2a18919ba56f16d89",
    (128, 1): "a9a7eef85227ed13475a84f4824fb81ef63299347ab44e75a71a140ad2bfa3aa",
    (256, 1): "ed7722d0e0a4b1cb148c2a134548d991e7e6e0a2c36145a322ba138267921c18",
    (64, 5): "c84b3e7af6401c645cadc8901d0d719cbd226b07bfe9e0c6d232e5dfaaa60390",
    (512, 5): "eaaa7206e16d8132c5a17ef9d0673dfa7134d9b38717039494bb69576834042c",
}
