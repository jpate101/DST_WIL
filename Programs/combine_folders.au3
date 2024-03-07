

Local $vVariable = 1

While 1
Sleep(50)




FileMove ( "P:\DST_WIL\programs\outputs(2)\out"& $vVariable &".mlt", "P:\DST_WIL\programs\outputs\out"& ($vVariable+8640) &".mlt")


$vVariable = $vVariable + 1

If $vVariable > 17280 Then ExitLoop

WEnd