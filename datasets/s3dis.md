# S3DIS

The dataset contains some bugs which can be found and corrected at those position.

/S3DIS/raw/Area_2/auditorium_1/auditorium_1.txt row 741099: ['0.577500', '-12.407500', '2.604000', '167.\x1000000', '175.000000', '177.000000']

/S3DIS/raw/Area_3/hallway_2/hallway_2.txt row 926335: ['19.302', '-9.1\x160', '1.785', '146', '137', '106']


```
diff --git a/Area_3/hallway_2/hallway_2.txt b/Area_3/hallway_2/hallway_2.txt
index 02f32b8..870566e 100644
--- a/Area_3/hallway_2/hallway_2.txt
+++ b/Area_3/hallway_2/hallway_2.txt
@@ -926334,7 +926334,7 @@
 19.237 -9.161 1.561 141 131 96
 19.248 -9.160 1.768 136 129 103
 19.276 -9.160 1.684 139 130 99
-19.302 -9.10 1.785 146 137 106
+19.302 -9.1 0 1.785 146 137 106
 19.242 -9.160 1.790 146 134 108
 19.271 -9.160 1.679 140 129 99
 19.278 -9.160 1.761 133 123 98
diff --git a/Area_5/hallway_6/Annotations/ceiling_1.txt b/Area_5/hallway_6/Annotations/ceiling_1.txt
index 62e563d..3a9087b 100644
--- a/Area_5/hallway_6/Annotations/ceiling_1.txt
+++ b/Area_5/hallway_6/Annotations/ceiling_1.txt
@@ -180386,7 +180386,7 @@
 22.383 6.858 3.050 155 155 165
 22.275 6.643 3.048 192 194 191
 22.359 6.835 3.050 152 152 162
-22.350 6.692 3.048 185187 182
+22.350 6.692 3.048 185 187 182
 22.314 6.638 3.048 170 171 175
 22.481 6.818 3.049 149 149 159
 22.328 6.673 3.048 190 195 191
diff --git a/Area_6/copyRoom_1/copy_Room_1.txt b/Area_6/copyRoom_1/copyRoom_1.txt
similarity index 100%
rename from Area_6/copyRoom_1/copy_Room_1.txt
rename to Area_6/copyRoom_1/copyRoom_1.txt

```