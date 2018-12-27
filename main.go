package main

import (
	"database/sql"
	"fmt"
	"image"
	"image/color"
	"log"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/Kagami/go-face"
	_ "github.com/lib/pq"
	"gocv.io/x/gocv"
)

const dataDir = "images"
const (
	host     = "localhost"
	port     = 5432
	user     = "postgres"
	password = "mysecretpassword"
	dbname   = "faces"
)

func main() {
	deviceID := 0

	psqlInfo := fmt.Sprintf("host=%s port=%d user=%s "+
		"password=%s dbname=%s sslmode=disable",
		host, port, user, password, dbname)

	// reader := bufio.NewReader(os.Stdin)
	// fmt.Print("Enter person name: ")
	// userName, _ := reader.ReadString('\n')

	// userName = strings.Replace(userName, "\n", "", -1)
	// cleanedName := strings.Replace(userName, " ", "", -1)
	// fmt.Println("Name: ", cleanedName)
	// var fileNameParts []string
	// fileNameParts = append(fileNameParts, cleanedName)
	// fileNameParts = append(fileNameParts, ".jpg")
	// saveFile := strings.Join(fileNameParts, "")
	// fmt.Println("saveFile: ", saveFile)

	var fileNameParts []string
	saveFile := time.Now().Format("20060102150405")
	fileNameParts = append(fileNameParts, saveFile, ".jpg")
	saveFile = strings.Join(fileNameParts, "")
	fmt.Println("saveFile: ", saveFile)

	// open display window
	window := gocv.NewWindow("Face Detect")
	defer window.Close()

	// color for the rect when faces detected
	blue := color.RGBA{0, 0, 255, 0}
	white := color.RGBA{255, 255, 255, 0}
	red := color.RGBA{255, 0, 0, 0}

	webcam, err := gocv.OpenVideoCapture(deviceID)
	if err != nil {
		fmt.Printf("Error opening video capture device: %v\n", deviceID)
		return
	}
	defer webcam.Close()

	time.Sleep(1 * time.Second) // allow time for camera to get light

	img := gocv.NewMat()
	defer img.Close()

	if ok := webcam.Read(&img); !ok {
		fmt.Printf("cannot read device %v\n", deviceID)
		return
	}
	if img.Empty() {
		fmt.Printf("no image on device %v\n", deviceID)
		return
	}

	gocv.IMWrite(saveFile, img)

	// load classifier to recognize faces
	classifier := gocv.NewCascadeClassifier()
	defer classifier.Close()

	if !classifier.Load("data/haarcascade_frontalface_default.xml") {
		fmt.Println("Error reading cascade file: data/haarcascade_frontalface_default.xml")
		return
	}

	// detect faces
	rects := classifier.DetectMultiScale(img)
	fmt.Printf("found %d face\n", len(rects))
	fmt.Print(rects)

	// draw a rectangle around each face on the original image
	for _, r := range rects {
		gocv.Rectangle(&img, r, blue, 1)
		nameTag := "Label"
		size := gocv.GetTextSize(nameTag, gocv.FontHersheyPlain, 1.2, 1)
		pt := image.Pt(r.Min.X+(r.Min.X/2)-(size.X/2), r.Max.Y+20)
		gocv.PutText(&img, nameTag, pt, gocv.FontHersheyPlain, 1.2, white, 2)
		gocv.Rectangle(&img, image.Rect(r.Min.X, r.Max.Y, r.Max.X, r.Max.Y+35), red, 1)
	}
	window.IMShow(img)

	//faces, err := rec.Recognize(img.ToBytes())

	var parts []string
	parts = append(parts, dataDir, "/", saveFile)
	filetowrite := strings.Join(parts, "")
	gocv.IMWrite(filetowrite, img)

	// Move file
	err = os.Rename(saveFile, filetowrite)
	if err != nil {
		log.Fatal(err)
	}

	// avengersImage := filepath.Join(dataDir, saveFile)

	rec, err := face.NewRecognizer(dataDir)
	if err != nil {
		fmt.Println("Cannot initialize recognizer", err)
		return
	}
	defer rec.Close()

	fmt.Println("Recognizer Initialized")

	faces, err := rec.RecognizeFile(filetowrite)

	if err != nil {
		log.Fatalf("Can't recognize: %v", err)
	}
	fmt.Println("Number of Faces in Image: ", len(faces))
	var samples []face.Descriptor
	for _, f := range faces {
		samples = append(samples, f.Descriptor)
	}

	var faceData []string
	for _, f := range samples[0] {
		faceData = append(faceData, fmt.Sprintf("%f", f))
	}

	faceDataForQuery := strings.Join(faceData, ",")
	//fmt.Println(faceDataForQuery)

	window.WaitKey(1000)

	// DB
	db, err := sql.Open("postgres", psqlInfo)
	if err != nil {
		panic(err)
	}
	defer db.Close()

	err = db.Ping()
	if err != nil {
		panic(err)
	}
	fmt.Println("Successfully connected to db!")

	sqlStatement := `
	INSERT INTO face.vectors(filename, vector)
	VALUES ($1, $2);`
	_, err = db.Exec(sqlStatement, saveFile, faceDataForQuery)
	if err != nil {
		panic(err)
	}
	fmt.Println("Successfully created record in db!")

	getAllDataFromDb()

	// http.HandleFunc("/", sayHello)
	// if err := http.ListenAndServe(":8020", nil); err != nil {
	// 	panic(err)
	// }
}

func sayHello(w http.ResponseWriter, r *http.Request) {
	message := r.URL.Path
	message = strings.TrimPrefix(message, "/")
	message = "Hello " + message
	w.Write([]byte(message))
	fmt.Println("ping")
}

func getAllDataFromDb() {
	var (
		filename string
		vector   []uint8
	)
	psqlInfo := fmt.Sprintf("host=%s port=%d user=%s "+
		"password=%s dbname=%s sslmode=disable",
		host, port, user, password, dbname)
	db, err := sql.Open("postgres", psqlInfo)
	if err != nil {
		panic(err)
	}
	defer db.Close()

	err = db.Ping()
	if err != nil {
		panic(err)
	}
	rows, err := db.Query("select filename, CUBE(vector) from face.vectors")
	if err != nil {
		log.Fatal(err)
	}
	defer rows.Close()
	for rows.Next() {
		err := rows.Scan(&filename, &vector)
		if err != nil {
			log.Fatal(err)
		}
		log.Println(filename, vector)
	}
	err = rows.Err()
	if err != nil {
		log.Fatal(err)
	}
}
