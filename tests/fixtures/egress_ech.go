// Send a TLS 1.3 encrypted ClientHello to the fixture proxy's HTTPS listener.
package main

import (
	"crypto/ecdh"
	"crypto/rand"
	"crypto/tls"
	"encoding/binary"
	"fmt"
	"net"
	"os"
	"time"
)

func vector(data []byte) []byte {
	return append(binary.BigEndian.AppendUint16(nil, uint16(len(data))), data...)
}

func main() {
	key, err := ecdh.X25519().GenerateKey(rand.Reader)
	if err != nil {
		panic(err)
	}
	// RFC 9849 ECHConfig: X25519, HKDF-SHA256, AES-128-GCM, public SNI.
	config := append([]byte{1, 0, 32}, vector(key.PublicKey().Bytes())...)
	config = append(config, vector([]byte{0, 1, 0, 1})...)
	name := "full.test"
	config = append(config, 0, byte(len(name)))
	config = append(config, name...)
	config = append(config, 0, 0)
	list := vector(append([]byte{0xfe, 0x0d}, vector(config)...))
	conn, err := tls.DialWithDialer(&net.Dialer{Timeout: 5 * time.Second}, "tcp", "full.test:443", &tls.Config{
		MinVersion: tls.VersionTLS13, ServerName: "private.test", EncryptedClientHelloConfigList: list,
	})
	if err == nil {
		conn.Close()
		fmt.Fprintln(os.Stderr, "proxy accepted the encrypted ClientHello")
		os.Exit(1)
	}
}
